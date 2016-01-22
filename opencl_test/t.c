//====== OPENCL KERNEL START
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define GROUP_SIZE 128

#define CALC_2D_IDX(M1,M2,m1,m2) (((m1)*(M2))+((m2)))
#define CALC_3D_IDX(M1,M2,M3,m1,m2,m3) (((m1)*(M2)*(M3))+((m2)*(M3))+((m3)))
#define CALC_4D_IDX(M1,M2,M3,M4,m1,m2,m3,m4) (((m1)*(M2)*(M3)*(M4))+((m2)*(M3)*(M4))+((m3)*(M4))+((m4)))
#define CALC_5D_IDX(M1,M2,M3,M4,M5,m1,m2,m3,m4,m5) (((m1)*(M2)*(M3)*(M4)*(M5))+((m2)*(M3)*(M4)*(M5))+((m3)*(M4)*(M5))+((m4)*(M5))+((m5)))
#define CALC_6D_IDX(M1,M2,M3,M4,M5,M6,m1,m2,m3,m4,m5,m6) (((m1)*(M2)*(M3)*(M4)*(M5)*(M6))+((m2)*(M3)*(M4)*(M5)*(M6))+((m3)*(M4)*(M5)*(M6))+((m4)*(M5)*(M6))+((m5)*(M6))+((m6)))
#define CALC_7D_IDX(M1,M2,M3,M4,M5,M6,M7,m1,m2,m3,m4,m5,m6,m7) (((m1)*(M2)*(M3)*(M4)*(M5)*(M6)*(M7))+((m2)*(M3)*(M4)*(M5)*(M6)*(M7))+((m3)*(M4)*(M5)*(M6)*(M7))+((m4)*(M5)*(M6)*(M7))+((m5)*(M6)*(M7))+((m6)*(M7))+((m7)))
#define CALC_8D_IDX(M1,M2,M3,M4,M5,M6,M7,M8,m1,m2,m3,m4,m5,m6,m7,m8) (((m1)*(M2)*(M3)*(M4)*(M5)*(M6)*(M7)*(M8))+((m2)*(M3)*(M4)*(M5)*(M6)*(M7)*(M8))+((m3)*(M4)*(M5)*(M6)*(M7)*(M8))+((m4)*(M5)*(M6)*(M7)*(M8))+((m5)*(M6)*(M7)*(M8))+((m6)*(M7)*(M8))+((m7)*(M8))+((m8)))

#define __global
#define __kernel
#define __constant

static void exact_solution_g4(double xi, double eta, double zeta,
			      double dtemp[5], __global double *ce);
static void binvcrhs_p0_p1_p2(__global double *lhs, __global double *c,
			      __global double *r, unsigned arg_0_offset,
			      unsigned arg_1_offset, unsigned arg_2_offset);
static void matvec_sub_p0_p1_p2(__global double *ablock, __global double *avec,
				__global double *bvec, unsigned arg_0_offset,
				unsigned arg_1_offset, unsigned arg_2_offset);
static void matmul_sub_p0_p1_p2(__global double *ablock,
				__global double *bblock,
				__global double *cblock, unsigned arg_0_offset,
				unsigned arg_1_offset, unsigned arg_2_offset);
static void binvrhs_p0_p1(__global double *lhs, __global double *r,
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

static void exact_solution_g4(double xi, double eta, double zeta,
			      double dtemp[5], __global double *ce)
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

static void binvcrhs_p0_p1_p2(__global double *lhs, __global double *c,
			      __global double *r, unsigned arg_0_offset,
			      unsigned arg_1_offset, unsigned arg_2_offset)
{
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

static void matvec_sub_p0_p1_p2(__global double *ablock, __global double *avec,
				__global double *bvec, unsigned arg_0_offset,
				unsigned arg_1_offset, unsigned arg_2_offset)
{
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

static void matmul_sub_p0_p1_p2(__global double *ablock,
				__global double *bblock,
				__global double *cblock, unsigned arg_0_offset,
				unsigned arg_1_offset, unsigned arg_2_offset)
{
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

static void binvrhs_p0_p1(__global double *lhs, __global double *r,
			  unsigned arg_0_offset, unsigned arg_1_offset)
{
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

// The original loop is defined at line: 200 of bt.c
// The nested loops are swaped. 
__kernel void add_0(__global double *u, __global double *rhs, int __ocl_k_bound,
		    int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int m;			/* Defined at bt.c : 196 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (m = 0; m < 5; m++) {
			u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] +
			    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))];
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 322 of bt.c
// The nested loops are swaped. 
__kernel void exact_rhs_0(__global double *forcing, int __ocl_k_bound,
			  int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int m;			/* Defined at bt.c : 316 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (m = 0; m < 5; m++) {
			forcing[CALC_4D_IDX(13, 13, 13, 6, (i), (j), (k), (m))]
			    = 0.0;
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 336 of bt.c
__kernel void exact_rhs_1(double dnym1, __global int *grid_points, double dnzm1,
			  double dnxm1, __global double *ue,
			  __global double *buf, __global double *cuf,
			  __global double *q, __global double *forcing,
			  double tx2, double dx1tx1, double c2, double xxcon1,
			  double dx2tx1, double xxcon2, double dx3tx1,
			  double dx4tx1, double c1, double xxcon3,
			  double xxcon4, double xxcon5, double dx5tx1,
			  double dssp, __global double *ce, int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int j = get_global_id(0) + 1;
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double eta;		/* Defined at bt.c : 315 */
	int k;			/* Defined at bt.c : 316 */
	double zeta;		/* Defined at bt.c : 315 */
	int i;			/* Defined at bt.c : 316 */
	double xi;		/* Defined at bt.c : 315 */
	double dtemp[5];	/* Defined at bt.c : 315 */
	int m;			/* Defined at bt.c : 316 */
	double dtpp;		/* Defined at bt.c : 315 */
	int im1;		/* Defined at bt.c : 316 */
	int ip1;		/* Defined at bt.c : 316 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		eta = (double)j *dnym1;
		for (k = 1; k < grid_points[(2)] - 1; k++) {
			zeta = (double)k *dnzm1;
			for (i = 0; i < grid_points[(0)]; i++) {
				xi = (double)i *dnxm1;
				exact_solution_g4(xi, eta, zeta, dtemp,
						  ce) /*Arg Expension: ce */ ;
				for (m = 0; m < 5; m++) {
					ue[CALC_2D_IDX(12, 5, (i), (m))] =
					    dtemp[m];
				}
				dtpp = 1.0 / dtemp[0];
				for (m = 1; m <= 4; m++) {
					buf[CALC_2D_IDX(12, 5, (i), (m))] =
					    dtpp * dtemp[m];
				}
				cuf[(i)] =
				    buf[CALC_2D_IDX(12, 5, (i), (1))] *
				    buf[CALC_2D_IDX(12, 5, (i), (1))];
				buf[CALC_2D_IDX(12, 5, (i), (0))] =
				    cuf[(i)] +
				    buf[CALC_2D_IDX(12, 5, (i), (2))] *
				    buf[CALC_2D_IDX(12, 5, (i), (2))] +
				    buf[CALC_2D_IDX(12, 5, (i), (3))] *
				    buf[CALC_2D_IDX(12, 5, (i), (3))];
				q[(i)] =
				    0.5 * (buf[CALC_2D_IDX(12, 5, (i), (1))] *
					   ue[CALC_2D_IDX(12, 5, (i), (1))] +
					   buf[CALC_2D_IDX(12, 5, (i), (2))] *
					   ue[CALC_2D_IDX(12, 5, (i), (2))] +
					   buf[CALC_2D_IDX(12, 5, (i), (3))] *
					   ue[CALC_2D_IDX(12, 5, (i), (3))]);
			}
			for (i = 1; i < grid_points[(0)] - 1; i++) {
				im1 = i - 1;
				ip1 = i + 1;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (0))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (0))] - tx2 * (ue[CALC_2D_IDX(12,
									   5,
									   (ip1),
									   (1))]
							    -
							    ue[CALC_2D_IDX
							       (12, 5, (im1),
								(1))]) +
				    dx1tx1 *
				    (ue[CALC_2D_IDX(12, 5, (ip1), (0))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (i), (0))] +
				     ue[CALC_2D_IDX(12, 5, (im1), (0))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (1))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (1))] -
				    tx2 *
				    ((ue[CALC_2D_IDX(12, 5, (ip1), (1))] *
				      buf[CALC_2D_IDX(12, 5, (ip1), (1))] +
				      c2 * (ue[CALC_2D_IDX(12, 5, (ip1), (4))] -
					    q[(ip1)])) - (ue[CALC_2D_IDX(12, 5,
									 (im1),
									 (1))] *
							  buf[CALC_2D_IDX
							      (12, 5, (im1),
							       (1))] +
							  c2 *
							  (ue
							   [CALC_2D_IDX
							    (12, 5, (im1),
							     (4))] -
							   q[(im1)]))) +
				    xxcon1 *
				    (buf[CALC_2D_IDX(12, 5, (ip1), (1))] -
				     2.0 * buf[CALC_2D_IDX(12, 5, (i), (1))] +
				     buf[CALC_2D_IDX(12, 5, (im1), (1))]) +
				    dx2tx1 *
				    (ue[CALC_2D_IDX(12, 5, (ip1), (1))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (i), (1))] +
				     ue[CALC_2D_IDX(12, 5, (im1), (1))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (2))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (2))] - tx2 * (ue[CALC_2D_IDX(12,
									   5,
									   (ip1),
									   (2))]
							    *
							    buf[CALC_2D_IDX
								(12, 5, (ip1),
								 (1))] -
							    ue[CALC_2D_IDX
							       (12, 5, (im1),
								(2))] *
							    buf[CALC_2D_IDX
								(12, 5, (im1),
								 (1))]) +
				    xxcon2 *
				    (buf[CALC_2D_IDX(12, 5, (ip1), (2))] -
				     2.0 * buf[CALC_2D_IDX(12, 5, (i), (2))] +
				     buf[CALC_2D_IDX(12, 5, (im1), (2))]) +
				    dx3tx1 *
				    (ue[CALC_2D_IDX(12, 5, (ip1), (2))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (i), (2))] +
				     ue[CALC_2D_IDX(12, 5, (im1), (2))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (3))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (3))] - tx2 * (ue[CALC_2D_IDX(12,
									   5,
									   (ip1),
									   (3))]
							    *
							    buf[CALC_2D_IDX
								(12, 5, (ip1),
								 (1))] -
							    ue[CALC_2D_IDX
							       (12, 5, (im1),
								(3))] *
							    buf[CALC_2D_IDX
								(12, 5, (im1),
								 (1))]) +
				    xxcon2 *
				    (buf[CALC_2D_IDX(12, 5, (ip1), (3))] -
				     2.0 * buf[CALC_2D_IDX(12, 5, (i), (3))] +
				     buf[CALC_2D_IDX(12, 5, (im1), (3))]) +
				    dx4tx1 *
				    (ue[CALC_2D_IDX(12, 5, (ip1), (3))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (i), (3))] +
				     ue[CALC_2D_IDX(12, 5, (im1), (3))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (4))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (4))] - tx2 * (buf[CALC_2D_IDX(12,
									    5,
									    (ip1),
									    (1))]
							    * (c1 *
							       ue[CALC_2D_IDX
								  (12, 5, (ip1),
								   (4))] -
							       c2 * q[(ip1)]) -
							    buf[CALC_2D_IDX
								(12, 5, (im1),
								 (1))] * (c1 *
									  ue
									  [CALC_2D_IDX
									   (12,
									    5,
									    (im1),
									    (4))]
									  -
									  c2 *
									  q[(im1)])) + 0.5 * xxcon3 * (buf[CALC_2D_IDX(12, 5, (ip1), (0))] - 2.0 * buf[CALC_2D_IDX(12, 5, (i), (0))] + buf[CALC_2D_IDX(12, 5, (im1), (0))]) + xxcon4 * (cuf[(ip1)] - 2.0 * cuf[(i)] + cuf[(im1)]) + xxcon5 * (buf[CALC_2D_IDX(12, 5, (ip1), (4))] - 2.0 * buf[CALC_2D_IDX(12, 5, (i), (4))] + buf[CALC_2D_IDX(12, 5, (im1), (4))]) + dx5tx1 * (ue[CALC_2D_IDX(12, 5, (ip1), (4))] - 2.0 * ue[CALC_2D_IDX(12, 5, (i), (4))] + ue[CALC_2D_IDX(12, 5, (im1), (4))]);
			}
			for (m = 0; m < 5; m++) {
				i = 1;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] -
				    dssp * (5.0 *
					    ue[CALC_2D_IDX(12, 5, (i), (m))] -
					    4.0 *
					    ue[CALC_2D_IDX(12, 5, (i + 1), (m))]
					    +
					    ue[CALC_2D_IDX
					       (12, 5, (i + 2), (m))]);
				i = 2;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] -
				    dssp * (-4.0 *
					    ue[CALC_2D_IDX(12, 5, (i - 1), (m))]
					    +
					    6.0 *
					    ue[CALC_2D_IDX(12, 5, (i), (m))] -
					    4.0 *
					    ue[CALC_2D_IDX(12, 5, (i + 1), (m))]
					    +
					    ue[CALC_2D_IDX
					       (12, 5, (i + 2), (m))]);
			}
			for (m = 0; m < 5; m++) {
				for (i = 1 * 3;
				     i <= grid_points[(0)] - 3 * 1 - 1; i++) {
					forcing[CALC_4D_IDX
						(13, 13, 13, 6, (i), (j), (k),
						 (m))] =
					    forcing[CALC_4D_IDX
						    (13, 13, 13, 6, (i), (j),
						     (k),
						     (m))] -
					    dssp *
					    (ue
					     [CALC_2D_IDX(12, 5, (i - 2), (m))]
					     -
					     4.0 *
					     ue[CALC_2D_IDX
						(12, 5, (i - 1),
						 (m))] +
					     6.0 *
					     ue[CALC_2D_IDX(12, 5, (i), (m))] -
					     4.0 *
					     ue[CALC_2D_IDX
						(12, 5, (i + 1),
						 (m))] + ue[CALC_2D_IDX(12, 5,
									(i + 2),
									(m))]);
				}
			}
			for (m = 0; m < 5; m++) {
				i = grid_points[(0)] - 3;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] - dssp * (ue[CALC_2D_IDX(12,
									    5,
									    (i -
									     2),
									    (m))]
							     -
							     4.0 *
							     ue[CALC_2D_IDX
								(12, 5, (i - 1),
								 (m))] +
							     6.0 *
							     ue[CALC_2D_IDX
								(12, 5, (i),
								 (m))] -
							     4.0 *
							     ue[CALC_2D_IDX
								(12, 5, (i + 1),
								 (m))]);
				i = grid_points[(0)] - 2;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] - dssp * (ue[CALC_2D_IDX(12,
									    5,
									    (i -
									     2),
									    (m))]
							     -
							     4.0 *
							     ue[CALC_2D_IDX
								(12, 5, (i - 1),
								 (m))] +
							     5.0 *
							     ue[CALC_2D_IDX
								(12, 5, (i),
								 (m))]);
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 437 of bt.c
__kernel void exact_rhs_2(double dnxm1, __global int *grid_points, double dnzm1,
			  double dnym1, __global double *ue,
			  __global double *buf, __global double *cuf,
			  __global double *q, __global double *forcing,
			  double ty2, double dy1ty1, double yycon2,
			  double dy2ty1, double c2, double yycon1,
			  double dy3ty1, double dy4ty1, double c1,
			  double yycon3, double yycon4, double yycon5,
			  double dy5ty1, double dssp, __global double *ce,
			  int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int i = get_global_id(0) + 1;
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double xi;		/* Defined at bt.c : 315 */
	int k;			/* Defined at bt.c : 316 */
	double zeta;		/* Defined at bt.c : 315 */
	int j;			/* Defined at bt.c : 316 */
	double eta;		/* Defined at bt.c : 315 */
	double dtemp[5];	/* Defined at bt.c : 315 */
	int m;			/* Defined at bt.c : 316 */
	double dtpp;		/* Defined at bt.c : 315 */
	int jm1;		/* Defined at bt.c : 316 */
	int jp1;		/* Defined at bt.c : 316 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		xi = (double)i *dnxm1;
		for (k = 1; k < grid_points[(2)] - 1; k++) {
			zeta = (double)k *dnzm1;
			for (j = 0; j < grid_points[(1)]; j++) {
				eta = (double)j *dnym1;
				exact_solution_g4(xi, eta, zeta, dtemp,
						  ce) /*Arg Expension: ce */ ;
				for (m = 0; m < 5; m++) {
					ue[CALC_2D_IDX(12, 5, (j), (m))] =
					    dtemp[m];
				}
				dtpp = 1.0 / dtemp[0];
				for (m = 1; m <= 4; m++) {
					buf[CALC_2D_IDX(12, 5, (j), (m))] =
					    dtpp * dtemp[m];
				}
				cuf[(j)] =
				    buf[CALC_2D_IDX(12, 5, (j), (2))] *
				    buf[CALC_2D_IDX(12, 5, (j), (2))];
				buf[CALC_2D_IDX(12, 5, (j), (0))] =
				    cuf[(j)] +
				    buf[CALC_2D_IDX(12, 5, (j), (1))] *
				    buf[CALC_2D_IDX(12, 5, (j), (1))] +
				    buf[CALC_2D_IDX(12, 5, (j), (3))] *
				    buf[CALC_2D_IDX(12, 5, (j), (3))];
				q[(j)] =
				    0.5 * (buf[CALC_2D_IDX(12, 5, (j), (1))] *
					   ue[CALC_2D_IDX(12, 5, (j), (1))] +
					   buf[CALC_2D_IDX(12, 5, (j), (2))] *
					   ue[CALC_2D_IDX(12, 5, (j), (2))] +
					   buf[CALC_2D_IDX(12, 5, (j), (3))] *
					   ue[CALC_2D_IDX(12, 5, (j), (3))]);
			}
			for (j = 1; j < grid_points[(1)] - 1; j++) {
				jm1 = j - 1;
				jp1 = j + 1;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (0))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (0))] - ty2 * (ue[CALC_2D_IDX(12,
									   5,
									   (jp1),
									   (2))]
							    -
							    ue[CALC_2D_IDX
							       (12, 5, (jm1),
								(2))]) +
				    dy1ty1 *
				    (ue[CALC_2D_IDX(12, 5, (jp1), (0))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (j), (0))] +
				     ue[CALC_2D_IDX(12, 5, (jm1), (0))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (1))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (1))] - ty2 * (ue[CALC_2D_IDX(12,
									   5,
									   (jp1),
									   (1))]
							    *
							    buf[CALC_2D_IDX
								(12, 5, (jp1),
								 (2))] -
							    ue[CALC_2D_IDX
							       (12, 5, (jm1),
								(1))] *
							    buf[CALC_2D_IDX
								(12, 5, (jm1),
								 (2))]) +
				    yycon2 *
				    (buf[CALC_2D_IDX(12, 5, (jp1), (1))] -
				     2.0 * buf[CALC_2D_IDX(12, 5, (j), (1))] +
				     buf[CALC_2D_IDX(12, 5, (jm1), (1))]) +
				    dy2ty1 *
				    (ue[CALC_2D_IDX(12, 5, (jp1), (1))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (j), (1))] +
				     ue[CALC_2D_IDX(12, 5, (jm1), (1))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (2))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (2))] -
				    ty2 *
				    ((ue[CALC_2D_IDX(12, 5, (jp1), (2))] *
				      buf[CALC_2D_IDX(12, 5, (jp1), (2))] +
				      c2 * (ue[CALC_2D_IDX(12, 5, (jp1), (4))] -
					    q[(jp1)])) - (ue[CALC_2D_IDX(12, 5,
									 (jm1),
									 (2))] *
							  buf[CALC_2D_IDX
							      (12, 5, (jm1),
							       (2))] +
							  c2 *
							  (ue
							   [CALC_2D_IDX
							    (12, 5, (jm1),
							     (4))] -
							   q[(jm1)]))) +
				    yycon1 *
				    (buf[CALC_2D_IDX(12, 5, (jp1), (2))] -
				     2.0 * buf[CALC_2D_IDX(12, 5, (j), (2))] +
				     buf[CALC_2D_IDX(12, 5, (jm1), (2))]) +
				    dy3ty1 *
				    (ue[CALC_2D_IDX(12, 5, (jp1), (2))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (j), (2))] +
				     ue[CALC_2D_IDX(12, 5, (jm1), (2))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (3))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (3))] - ty2 * (ue[CALC_2D_IDX(12,
									   5,
									   (jp1),
									   (3))]
							    *
							    buf[CALC_2D_IDX
								(12, 5, (jp1),
								 (2))] -
							    ue[CALC_2D_IDX
							       (12, 5, (jm1),
								(3))] *
							    buf[CALC_2D_IDX
								(12, 5, (jm1),
								 (2))]) +
				    yycon2 *
				    (buf[CALC_2D_IDX(12, 5, (jp1), (3))] -
				     2.0 * buf[CALC_2D_IDX(12, 5, (j), (3))] +
				     buf[CALC_2D_IDX(12, 5, (jm1), (3))]) +
				    dy4ty1 *
				    (ue[CALC_2D_IDX(12, 5, (jp1), (3))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (j), (3))] +
				     ue[CALC_2D_IDX(12, 5, (jm1), (3))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (4))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (4))] - ty2 * (buf[CALC_2D_IDX(12,
									    5,
									    (jp1),
									    (2))]
							    * (c1 *
							       ue[CALC_2D_IDX
								  (12, 5, (jp1),
								   (4))] -
							       c2 * q[(jp1)]) -
							    buf[CALC_2D_IDX
								(12, 5, (jm1),
								 (2))] * (c1 *
									  ue
									  [CALC_2D_IDX
									   (12,
									    5,
									    (jm1),
									    (4))]
									  -
									  c2 *
									  q[(jm1)])) + 0.5 * yycon3 * (buf[CALC_2D_IDX(12, 5, (jp1), (0))] - 2.0 * buf[CALC_2D_IDX(12, 5, (j), (0))] + buf[CALC_2D_IDX(12, 5, (jm1), (0))]) + yycon4 * (cuf[(jp1)] - 2.0 * cuf[(j)] + cuf[(jm1)]) + yycon5 * (buf[CALC_2D_IDX(12, 5, (jp1), (4))] - 2.0 * buf[CALC_2D_IDX(12, 5, (j), (4))] + buf[CALC_2D_IDX(12, 5, (jm1), (4))]) + dy5ty1 * (ue[CALC_2D_IDX(12, 5, (jp1), (4))] - 2.0 * ue[CALC_2D_IDX(12, 5, (j), (4))] + ue[CALC_2D_IDX(12, 5, (jm1), (4))]);
			}
			for (m = 0; m < 5; m++) {
				j = 1;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] -
				    dssp * (5.0 *
					    ue[CALC_2D_IDX(12, 5, (j), (m))] -
					    4.0 *
					    ue[CALC_2D_IDX(12, 5, (j + 1), (m))]
					    +
					    ue[CALC_2D_IDX
					       (12, 5, (j + 2), (m))]);
				j = 2;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] -
				    dssp * (-4.0 *
					    ue[CALC_2D_IDX(12, 5, (j - 1), (m))]
					    +
					    6.0 *
					    ue[CALC_2D_IDX(12, 5, (j), (m))] -
					    4.0 *
					    ue[CALC_2D_IDX(12, 5, (j + 1), (m))]
					    +
					    ue[CALC_2D_IDX
					       (12, 5, (j + 2), (m))]);
			}
			for (m = 0; m < 5; m++) {
				for (j = 1 * 3;
				     j <= grid_points[(1)] - 3 * 1 - 1; j++) {
					forcing[CALC_4D_IDX
						(13, 13, 13, 6, (i), (j), (k),
						 (m))] =
					    forcing[CALC_4D_IDX
						    (13, 13, 13, 6, (i), (j),
						     (k),
						     (m))] -
					    dssp *
					    (ue
					     [CALC_2D_IDX(12, 5, (j - 2), (m))]
					     -
					     4.0 *
					     ue[CALC_2D_IDX
						(12, 5, (j - 1),
						 (m))] +
					     6.0 *
					     ue[CALC_2D_IDX(12, 5, (j), (m))] -
					     4.0 *
					     ue[CALC_2D_IDX
						(12, 5, (j + 1),
						 (m))] + ue[CALC_2D_IDX(12, 5,
									(j + 2),
									(m))]);
				}
			}
			for (m = 0; m < 5; m++) {
				j = grid_points[(1)] - 3;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] - dssp * (ue[CALC_2D_IDX(12,
									    5,
									    (j -
									     2),
									    (m))]
							     -
							     4.0 *
							     ue[CALC_2D_IDX
								(12, 5, (j - 1),
								 (m))] +
							     6.0 *
							     ue[CALC_2D_IDX
								(12, 5, (j),
								 (m))] -
							     4.0 *
							     ue[CALC_2D_IDX
								(12, 5, (j + 1),
								 (m))]);
				j = grid_points[(1)] - 2;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] - dssp * (ue[CALC_2D_IDX(12,
									    5,
									    (j -
									     2),
									    (m))]
							     -
							     4.0 *
							     ue[CALC_2D_IDX
								(12, 5, (j - 1),
								 (m))] +
							     5.0 *
							     ue[CALC_2D_IDX
								(12, 5, (j),
								 (m))]);
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 539 of bt.c
__kernel void exact_rhs_3(double dnxm1, __global int *grid_points, double dnym1,
			  double dnzm1, __global double *ue,
			  __global double *buf, __global double *cuf,
			  __global double *q, __global double *forcing,
			  double tz2, double dz1tz1, double zzcon2,
			  double dz2tz1, double dz3tz1, double c2,
			  double zzcon1, double dz4tz1, double c1,
			  double zzcon3, double zzcon4, double zzcon5,
			  double dz5tz1, double dssp, __global double *ce,
			  int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int i = get_global_id(0) + 1;
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double xi;		/* Defined at bt.c : 315 */
	int j;			/* Defined at bt.c : 316 */
	double eta;		/* Defined at bt.c : 315 */
	int k;			/* Defined at bt.c : 316 */
	double zeta;		/* Defined at bt.c : 315 */
	double dtemp[5];	/* Defined at bt.c : 315 */
	int m;			/* Defined at bt.c : 316 */
	double dtpp;		/* Defined at bt.c : 315 */
	int km1;		/* Defined at bt.c : 316 */
	int kp1;		/* Defined at bt.c : 316 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		xi = (double)i *dnxm1;
		for (j = 1; j < grid_points[(1)] - 1; j++) {
			eta = (double)j *dnym1;
			for (k = 0; k < grid_points[(2)]; k++) {
				zeta = (double)k *dnzm1;
				exact_solution_g4(xi, eta, zeta, dtemp,
						  ce) /*Arg Expension: ce */ ;
				for (m = 0; m < 5; m++) {
					ue[CALC_2D_IDX(12, 5, (k), (m))] =
					    dtemp[m];
				}
				dtpp = 1.0 / dtemp[0];
				for (m = 1; m <= 4; m++) {
					buf[CALC_2D_IDX(12, 5, (k), (m))] =
					    dtpp * dtemp[m];
				}
				cuf[(k)] =
				    buf[CALC_2D_IDX(12, 5, (k), (3))] *
				    buf[CALC_2D_IDX(12, 5, (k), (3))];
				buf[CALC_2D_IDX(12, 5, (k), (0))] =
				    cuf[(k)] +
				    buf[CALC_2D_IDX(12, 5, (k), (1))] *
				    buf[CALC_2D_IDX(12, 5, (k), (1))] +
				    buf[CALC_2D_IDX(12, 5, (k), (2))] *
				    buf[CALC_2D_IDX(12, 5, (k), (2))];
				q[(k)] =
				    0.5 * (buf[CALC_2D_IDX(12, 5, (k), (1))] *
					   ue[CALC_2D_IDX(12, 5, (k), (1))] +
					   buf[CALC_2D_IDX(12, 5, (k), (2))] *
					   ue[CALC_2D_IDX(12, 5, (k), (2))] +
					   buf[CALC_2D_IDX(12, 5, (k), (3))] *
					   ue[CALC_2D_IDX(12, 5, (k), (3))]);
			}
			for (k = 1; k < grid_points[(2)] - 1; k++) {
				km1 = k - 1;
				kp1 = k + 1;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (0))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (0))] - tz2 * (ue[CALC_2D_IDX(12,
									   5,
									   (kp1),
									   (3))]
							    -
							    ue[CALC_2D_IDX
							       (12, 5, (km1),
								(3))]) +
				    dz1tz1 *
				    (ue[CALC_2D_IDX(12, 5, (kp1), (0))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (k), (0))] +
				     ue[CALC_2D_IDX(12, 5, (km1), (0))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (1))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (1))] - tz2 * (ue[CALC_2D_IDX(12,
									   5,
									   (kp1),
									   (1))]
							    *
							    buf[CALC_2D_IDX
								(12, 5, (kp1),
								 (3))] -
							    ue[CALC_2D_IDX
							       (12, 5, (km1),
								(1))] *
							    buf[CALC_2D_IDX
								(12, 5, (km1),
								 (3))]) +
				    zzcon2 *
				    (buf[CALC_2D_IDX(12, 5, (kp1), (1))] -
				     2.0 * buf[CALC_2D_IDX(12, 5, (k), (1))] +
				     buf[CALC_2D_IDX(12, 5, (km1), (1))]) +
				    dz2tz1 *
				    (ue[CALC_2D_IDX(12, 5, (kp1), (1))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (k), (1))] +
				     ue[CALC_2D_IDX(12, 5, (km1), (1))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (2))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (2))] - tz2 * (ue[CALC_2D_IDX(12,
									   5,
									   (kp1),
									   (2))]
							    *
							    buf[CALC_2D_IDX
								(12, 5, (kp1),
								 (3))] -
							    ue[CALC_2D_IDX
							       (12, 5, (km1),
								(2))] *
							    buf[CALC_2D_IDX
								(12, 5, (km1),
								 (3))]) +
				    zzcon2 *
				    (buf[CALC_2D_IDX(12, 5, (kp1), (2))] -
				     2.0 * buf[CALC_2D_IDX(12, 5, (k), (2))] +
				     buf[CALC_2D_IDX(12, 5, (km1), (2))]) +
				    dz3tz1 *
				    (ue[CALC_2D_IDX(12, 5, (kp1), (2))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (k), (2))] +
				     ue[CALC_2D_IDX(12, 5, (km1), (2))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (3))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (3))] -
				    tz2 *
				    ((ue[CALC_2D_IDX(12, 5, (kp1), (3))] *
				      buf[CALC_2D_IDX(12, 5, (kp1), (3))] +
				      c2 * (ue[CALC_2D_IDX(12, 5, (kp1), (4))] -
					    q[(kp1)])) - (ue[CALC_2D_IDX(12, 5,
									 (km1),
									 (3))] *
							  buf[CALC_2D_IDX
							      (12, 5, (km1),
							       (3))] +
							  c2 *
							  (ue
							   [CALC_2D_IDX
							    (12, 5, (km1),
							     (4))] -
							   q[(km1)]))) +
				    zzcon1 *
				    (buf[CALC_2D_IDX(12, 5, (kp1), (3))] -
				     2.0 * buf[CALC_2D_IDX(12, 5, (k), (3))] +
				     buf[CALC_2D_IDX(12, 5, (km1), (3))]) +
				    dz4tz1 *
				    (ue[CALC_2D_IDX(12, 5, (kp1), (3))] -
				     2.0 * ue[CALC_2D_IDX(12, 5, (k), (3))] +
				     ue[CALC_2D_IDX(12, 5, (km1), (3))]);
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (4))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (4))] - tz2 * (buf[CALC_2D_IDX(12,
									    5,
									    (kp1),
									    (3))]
							    * (c1 *
							       ue[CALC_2D_IDX
								  (12, 5, (kp1),
								   (4))] -
							       c2 * q[(kp1)]) -
							    buf[CALC_2D_IDX
								(12, 5, (km1),
								 (3))] * (c1 *
									  ue
									  [CALC_2D_IDX
									   (12,
									    5,
									    (km1),
									    (4))]
									  -
									  c2 *
									  q[(km1)])) + 0.5 * zzcon3 * (buf[CALC_2D_IDX(12, 5, (kp1), (0))] - 2.0 * buf[CALC_2D_IDX(12, 5, (k), (0))] + buf[CALC_2D_IDX(12, 5, (km1), (0))]) + zzcon4 * (cuf[(kp1)] - 2.0 * cuf[(k)] + cuf[(km1)]) + zzcon5 * (buf[CALC_2D_IDX(12, 5, (kp1), (4))] - 2.0 * buf[CALC_2D_IDX(12, 5, (k), (4))] + buf[CALC_2D_IDX(12, 5, (km1), (4))]) + dz5tz1 * (ue[CALC_2D_IDX(12, 5, (kp1), (4))] - 2.0 * ue[CALC_2D_IDX(12, 5, (k), (4))] + ue[CALC_2D_IDX(12, 5, (km1), (4))]);
			}
			for (m = 0; m < 5; m++) {
				k = 1;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] -
				    dssp * (5.0 *
					    ue[CALC_2D_IDX(12, 5, (k), (m))] -
					    4.0 *
					    ue[CALC_2D_IDX(12, 5, (k + 1), (m))]
					    +
					    ue[CALC_2D_IDX
					       (12, 5, (k + 2), (m))]);
				k = 2;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] -
				    dssp * (-4.0 *
					    ue[CALC_2D_IDX(12, 5, (k - 1), (m))]
					    +
					    6.0 *
					    ue[CALC_2D_IDX(12, 5, (k), (m))] -
					    4.0 *
					    ue[CALC_2D_IDX(12, 5, (k + 1), (m))]
					    +
					    ue[CALC_2D_IDX
					       (12, 5, (k + 2), (m))]);
			}
			for (m = 0; m < 5; m++) {
				for (k = 1 * 3;
				     k <= grid_points[(2)] - 3 * 1 - 1; k++) {
					forcing[CALC_4D_IDX
						(13, 13, 13, 6, (i), (j), (k),
						 (m))] =
					    forcing[CALC_4D_IDX
						    (13, 13, 13, 6, (i), (j),
						     (k),
						     (m))] -
					    dssp *
					    (ue
					     [CALC_2D_IDX(12, 5, (k - 2), (m))]
					     -
					     4.0 *
					     ue[CALC_2D_IDX
						(12, 5, (k - 1),
						 (m))] +
					     6.0 *
					     ue[CALC_2D_IDX(12, 5, (k), (m))] -
					     4.0 *
					     ue[CALC_2D_IDX
						(12, 5, (k + 1),
						 (m))] + ue[CALC_2D_IDX(12, 5,
									(k + 2),
									(m))]);
				}
			}
			for (m = 0; m < 5; m++) {
				k = grid_points[(2)] - 3;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] - dssp * (ue[CALC_2D_IDX(12,
									    5,
									    (k -
									     2),
									    (m))]
							     -
							     4.0 *
							     ue[CALC_2D_IDX
								(12, 5, (k - 1),
								 (m))] +
							     6.0 *
							     ue[CALC_2D_IDX
								(12, 5, (k),
								 (m))] -
							     4.0 *
							     ue[CALC_2D_IDX
								(12, 5, (k + 1),
								 (m))]);
				k = grid_points[(2)] - 2;
				forcing[CALC_4D_IDX
					(13, 13, 13, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (13, 13, 13, 6, (i), (j), (k),
					     (m))] - dssp * (ue[CALC_2D_IDX(12,
									    5,
									    (k -
									     2),
									    (m))]
							     -
							     4.0 *
							     ue[CALC_2D_IDX
								(12, 5, (k - 1),
								 (m))] +
							     5.0 *
							     ue[CALC_2D_IDX
								(12, 5, (k),
								 (m))]);
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 640 of bt.c
// The nested loops are swaped. 
__kernel void exact_rhs_4(__global double *forcing, int __ocl_k_bound,
			  int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int m;			/* Defined at bt.c : 316 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (m = 0; m < 5; m++) {
			forcing[CALC_4D_IDX(13, 13, 13, 6, (i), (j), (k), (m))]
			    =
			    -1.0 *
			    forcing[CALC_4D_IDX
				    (13, 13, 13, 6, (i), (j), (k), (m))];
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 702 of bt.c
// The nested loops are swaped. 
__kernel void initialize_0(__global double *u)
{
	// Declare index variables (BEGIN)
	int k = get_global_id(0);
	int j = get_global_id(1);
	int i = get_global_id(2);
	if (!(k < 12)) {
		return;
	}
	if (!(j < 12)) {
		return;
	}
	if (!(i < 12)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int m;			/* Defined at bt.c : 691 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (m = 0; m < 5; m++) {
			u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] = 1.0;
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 719 of bt.c
__kernel void initialize_1(double dnxm1, __global int *grid_points,
			   double dnym1, double dnzm1, __global double *u,
			   __global double *ce, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int i = get_global_id(0);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double xi;		/* Defined at bt.c : 692 */
	int j;			/* Defined at bt.c : 691 */
	double eta;		/* Defined at bt.c : 692 */
	int k;			/* Defined at bt.c : 691 */
	double zeta;		/* Defined at bt.c : 692 */
	int ix;			/* Defined at bt.c : 691 */
	double Pface[2][3][5];	/* Defined at bt.c : 692 */
	int iy;			/* Defined at bt.c : 691 */
	int iz;			/* Defined at bt.c : 691 */
	int m;			/* Defined at bt.c : 691 */
	double Pxi;		/* Defined at bt.c : 692 */
	double Peta;		/* Defined at bt.c : 692 */
	double Pzeta;		/* Defined at bt.c : 692 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		xi = (double)i *dnxm1;
		for (j = 0; j < grid_points[(1)]; j++) {
			eta = (double)j *dnym1;
			for (k = 0; k < grid_points[(2)]; k++) {
				zeta = (double)k *dnzm1;
				for (ix = 0; ix < 2; ix++) {
					exact_solution_g4((double)ix, eta, zeta,
							  &(Pface[ix][0][0]),
							  ce)
					    /*Arg Expension: ce */ ;
				}
				for (iy = 0; iy < 2; iy++) {
					exact_solution_g4(xi, (double)iy, zeta,
							  &Pface[iy][1][0],
							  ce)
					    /*Arg Expension: ce */ ;
				}
				for (iz = 0; iz < 2; iz++) {
					exact_solution_g4(xi, eta, (double)iz,
							  &Pface[iz][2][0],
							  ce)
					    /*Arg Expension: ce */ ;
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
					  (13, 13, 13, 5, (i), (j), (k), (m))] =
				 Pxi + Peta + Pzeta - Pxi * Peta - Pxi * Pzeta -
				 Peta * Pzeta + Pxi * Peta * Pzeta;
				}
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 771 of bt.c
__kernel void initialize_2(double dnym1, __global int *grid_points,
			   double dnzm1, double xi, __global double *u, int i,
			   __global double *ce, int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int j = get_global_id(0);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double eta;		/* Defined at bt.c : 692 */
	int k;			/* Defined at bt.c : 691 */
	double zeta;		/* Defined at bt.c : 692 */
	double temp[5];		/* Defined at bt.c : 692 */
	int m;			/* Defined at bt.c : 691 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		eta = (double)j *dnym1;
		for (k = 0; k < grid_points[(2)]; k++) {
			zeta = (double)k *dnzm1;
			exact_solution_g4(xi, eta, zeta, temp,
					  ce) /*Arg Expension: ce */ ;
			for (m = 0; m < 5; m++) {
				u[CALC_4D_IDX
				  (13, 13, 13, 5, (i), (j), (k), (m))] =
			 temp[m];
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 791 of bt.c
__kernel void initialize_3(double dnym1, __global int *grid_points,
			   double dnzm1, double xi, __global double *u, int i,
			   __global double *ce, int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int j = get_global_id(0);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double eta;		/* Defined at bt.c : 692 */
	int k;			/* Defined at bt.c : 691 */
	double zeta;		/* Defined at bt.c : 692 */
	double temp[5];		/* Defined at bt.c : 692 */
	int m;			/* Defined at bt.c : 691 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		eta = (double)j *dnym1;
		for (k = 0; k < grid_points[(2)]; k++) {
			zeta = (double)k *dnzm1;
			exact_solution_g4(xi, eta, zeta, temp,
					  ce) /*Arg Expension: ce */ ;
			for (m = 0; m < 5; m++) {
				u[CALC_4D_IDX
				  (13, 13, 13, 5, (i), (j), (k), (m))] =
			 temp[m];
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 810 of bt.c
__kernel void initialize_4(double dnxm1, __global int *grid_points,
			   double dnzm1, double eta, __global double *u, int j,
			   __global double *ce, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int i = get_global_id(0);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double xi;		/* Defined at bt.c : 692 */
	int k;			/* Defined at bt.c : 691 */
	double zeta;		/* Defined at bt.c : 692 */
	double temp[5];		/* Defined at bt.c : 692 */
	int m;			/* Defined at bt.c : 691 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		xi = (double)i *dnxm1;
		for (k = 0; k < grid_points[(2)]; k++) {
			zeta = (double)k *dnzm1;
			exact_solution_g4(xi, eta, zeta, temp,
					  ce) /*Arg Expension: ce */ ;
			for (m = 0; m < 5; m++) {
				u[CALC_4D_IDX
				  (13, 13, 13, 5, (i), (j), (k), (m))] =
			 temp[m];
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 829 of bt.c
__kernel void initialize_5(double dnxm1, __global int *grid_points,
			   double dnzm1, double eta, __global double *u, int j,
			   __global double *ce, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int i = get_global_id(0);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double xi;		/* Defined at bt.c : 692 */
	int k;			/* Defined at bt.c : 691 */
	double zeta;		/* Defined at bt.c : 692 */
	double temp[5];		/* Defined at bt.c : 692 */
	int m;			/* Defined at bt.c : 691 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		xi = (double)i *dnxm1;
		for (k = 0; k < grid_points[(2)]; k++) {
			zeta = (double)k *dnzm1;
			exact_solution_g4(xi, eta, zeta, temp,
					  ce) /*Arg Expension: ce */ ;
			for (m = 0; m < 5; m++) {
				u[CALC_4D_IDX
				  (13, 13, 13, 5, (i), (j), (k), (m))] =
			 temp[m];
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 848 of bt.c
__kernel void initialize_6(double dnxm1, __global int *grid_points,
			   double dnym1, double zeta, __global double *u, int k,
			   __global double *ce, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int i = get_global_id(0);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double xi;		/* Defined at bt.c : 692 */
	int j;			/* Defined at bt.c : 691 */
	double eta;		/* Defined at bt.c : 692 */
	double temp[5];		/* Defined at bt.c : 692 */
	int m;			/* Defined at bt.c : 691 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		xi = (double)i *dnxm1;
		for (j = 0; j < grid_points[(1)]; j++) {
			eta = (double)j *dnym1;
			exact_solution_g4(xi, eta, zeta, temp,
					  ce) /*Arg Expension: ce */ ;
			for (m = 0; m < 5; m++) {
				u[CALC_4D_IDX
				  (13, 13, 13, 5, (i), (j), (k), (m))] =
			 temp[m];
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 867 of bt.c
__kernel void initialize_7(double dnxm1, __global int *grid_points,
			   double dnym1, double zeta, __global double *u, int k,
			   __global double *ce, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int i = get_global_id(0);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double xi;		/* Defined at bt.c : 692 */
	int j;			/* Defined at bt.c : 691 */
	double eta;		/* Defined at bt.c : 692 */
	double temp[5];		/* Defined at bt.c : 692 */
	int m;			/* Defined at bt.c : 691 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		xi = (double)i *dnxm1;
		for (j = 0; j < grid_points[(1)]; j++) {
			eta = (double)j *dnym1;
			exact_solution_g4(xi, eta, zeta, temp,
					  ce) /*Arg Expension: ce */ ;
			for (m = 0; m < 5; m++) {
				u[CALC_4D_IDX
				  (13, 13, 13, 5, (i), (j), (k), (m))] =
			 temp[m];
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 895 of bt.c
// The nested loops are swaped. 
__kernel void lhsinit_0(__global double *lhs, int __ocl_k_bound,
			int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int m;			/* Defined at bt.c : 885 */
	int n;			/* Defined at bt.c : 885 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (m = 0; m < 5; m++) {
			for (n = 0; n < 5; n++) {
				lhs[CALC_6D_IDX
				    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0),
				     (m), (n))] = 0.0;
				lhs[CALC_6D_IDX
				    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1),
				     (m), (n))] = 0.0;
				lhs[CALC_6D_IDX
				    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2),
				     (m), (n))] = 0.0;
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 915 of bt.c
// The nested loops are swaped. 
__kernel void lhsinit_1(__global double *lhs, int __ocl_k_bound,
			int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int m;			/* Defined at bt.c : 885 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (m = 0; m < 5; m++) {
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (m),
			     (m))] = 1.0;
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 946 of bt.c
// The nested loops are swaped. 
__kernel void lhsx_0(__global int *grid_points, double tmp1, __global double *u,
		     double tmp2, double tmp3, __global double *fjac, double c2,
		     double c1, __global double *njac, double con43,
		     double c3c4, double c1345, double dt, double tx1,
		     double tx2, __global double *lhs, double dx1, double dx2,
		     double dx3, double dx4, double dx5, int __ocl_k_bound,
		     int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int i;			/* Defined at bt.c : 939 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (i = 0; i < grid_points[(0)]; i++) {
			tmp1 =
			    1.0 /
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))];
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (0), (0))] = 0.0;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (0), (1))] = 1.0;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (0), (2))] = 0.0;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (0), (3))] = 0.0;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (0), (4))] = 0.0;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (1), (0))] =
			    -(u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))]
			      * tmp2 *
			      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))])
			    +
			    c2 * 0.50 *
			    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
			     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
			     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
			     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
			     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
			     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))])
			    * tmp2;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (1), (1))] =
			    (2.0 -
			     c2) *
			    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] /
			     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))]);
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (1), (2))] =
			    -c2 *
			    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
			     tmp1);
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (1), (3))] =
			    -c2 *
			    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
			     tmp1);
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (1), (4))] = c2;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (2), (0))] =
			    -(u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))]
			      *
			      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))])
			    * tmp2;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (2), (1))] =
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
			    tmp1;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (2), (2))] =
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
			    tmp1;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (2), (3))] = 0.0;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (2), (4))] = 0.0;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (3), (0))] =
			    -(u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))]
			      *
			      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))])
			    * tmp2;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (3), (1))] =
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
			    tmp1;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (3), (2))] = 0.0;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (3), (3))] =
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
			    tmp1;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (3), (4))] = 0.0;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (4), (0))] =
			    (c2 *
			     (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))]
			      *
			      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))]
			      +
			      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))]
			      *
			      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))]
			      +
			      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]
			      *
			      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))])
			     * tmp2 -
			     c1 *
			     (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))]
			      * tmp1)) * (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j),
							(k), (1))] * tmp1);
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (4), (1))] =
			    c1 *
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] *
			    tmp1 -
			    0.50 * c2 * (3.0 *
					 u[CALC_4D_IDX
					   (13, 13, 13, 5, (i), (j), (k),
					    (1))] * u[CALC_4D_IDX(13, 13, 13, 5,
								  (i), (j), (k),
								  (1))] +
					 u[CALC_4D_IDX
					   (13, 13, 13, 5, (i), (j), (k),
					    (2))] * u[CALC_4D_IDX(13, 13, 13, 5,
								  (i), (j), (k),
								  (2))] +
					 u[CALC_4D_IDX
					   (13, 13, 13, 5, (i), (j), (k),
					    (3))] * u[CALC_4D_IDX(13, 13, 13, 5,
								  (i), (j), (k),
								  (3))]) * tmp2;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (4), (2))] =
			    -c2 *
			    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
			     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))])
			    * tmp2;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (4), (3))] =
			    -c2 *
			    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
			     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))])
			    * tmp2;
			fjac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (4), (4))] =
			    c1 *
			    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
			     tmp1);
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (0), (0))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (0), (1))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (0), (2))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (0), (3))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (0), (4))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (1), (0))] =
			    -con43 * c3c4 * tmp2 *
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))];
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (1), (1))] =
			    con43 * c3c4 * tmp1;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (1), (2))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (1), (3))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (1), (4))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (2), (0))] =
			    -c3c4 * tmp2 *
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))];
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (2), (1))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (2), (2))] =
			    c3c4 * tmp1;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (2), (3))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (2), (4))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (3), (0))] =
			    -c3c4 * tmp2 *
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))];
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (3), (1))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (3), (2))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (3), (3))] =
			    c3c4 * tmp1;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (3), (4))] = 0.0;
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (4), (0))] =
			    -(con43 * c3c4 -
			      c1345) * tmp3 *
			    (((u
			       [CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))])
			      *
			      (u
			       [CALC_4D_IDX
				(13, 13, 13, 5, (i), (j), (k),
				 (1))]))) - (c3c4 -
					     c1345) * tmp3 *
			    (((u
			       [CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))])
			      *
			      (u
			       [CALC_4D_IDX
				(13, 13, 13, 5, (i), (j), (k),
				 (2))]))) - (c3c4 -
					     c1345) * tmp3 *
			    (((u
			       [CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))])
			      *
			      (u
			       [CALC_4D_IDX
				(13, 13, 13, 5, (i), (j), (k),
				 (3))]))) - c1345 * tmp2 * u[CALC_4D_IDX(13, 13,
									 13, 5,
									 (i),
									 (j),
									 (k),
									 (4))];
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (4), (1))] =
			    (con43 * c3c4 -
			     c1345) * tmp2 * u[CALC_4D_IDX(13, 13, 13, 5, (i),
							   (j), (k), (1))];
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (4), (2))] =
			    (c3c4 -
			     c1345) * tmp2 * u[CALC_4D_IDX(13, 13, 13, 5, (i),
							   (j), (k), (2))];
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (4), (3))] =
			    (c3c4 -
			     c1345) * tmp2 * u[CALC_4D_IDX(13, 13, 13, 5, (i),
							   (j), (k), (3))];
			njac[CALC_5D_IDX
			     (12, 12, 11, 5, 5, (i), (j), (k), (4), (4))] =
			    (c1345) * tmp1;
		}
		for (i = 1; i < grid_points[(0)] - 1; i++) {
			tmp1 = dt * tx1;
			tmp2 = dt * tx2;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0),
			     (0))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (0),
				  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (0),
								  (0))] -
			    tmp1 * dx1;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0),
			     (1))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (0),
				  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (0),
								  (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0),
			     (2))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (0),
				  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (0),
								  (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0),
			     (3))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (0),
				  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (0),
								  (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0),
			     (4))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (0),
				  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (0),
								  (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1),
			     (0))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (1),
				  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (1),
								  (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1),
			     (1))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (1),
				  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (1),
								  (1))] -
			    tmp1 * dx2;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1),
			     (2))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (1),
				  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (1),
								  (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1),
			     (3))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (1),
				  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (1),
								  (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1),
			     (4))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (1),
				  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (1),
								  (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2),
			     (0))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (2),
				  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (2),
								  (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2),
			     (1))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (2),
				  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (2),
								  (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2),
			     (2))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (2),
				  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (2),
								  (2))] -
			    tmp1 * dx3;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2),
			     (3))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (2),
				  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (2),
								  (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2),
			     (4))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (2),
				  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (2),
								  (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3),
			     (0))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (3),
				  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (3),
								  (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3),
			     (1))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (3),
				  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (3),
								  (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3),
			     (2))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (3),
				  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (3),
								  (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3),
			     (3))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (3),
				  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (3),
								  (3))] -
			    tmp1 * dx4;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3),
			     (4))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (3),
				  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (3),
								  (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4),
			     (0))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (4),
				  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (4),
								  (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4),
			     (1))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (4),
				  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (4),
								  (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4),
			     (2))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (4),
				  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (4),
								  (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4),
			     (3))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (4),
				  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (4),
								  (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4),
			     (4))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i - 1), (j), (k), (4),
				  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i - 1),
								  (j), (k), (4),
								  (4))] -
			    tmp1 * dx5;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0),
			     (0))] =
			    1.0 +
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (0),
				  (0))] + tmp1 * 2.0 * dx1;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0),
			     (1))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (0), (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0),
			     (2))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (0), (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0),
			     (3))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (0), (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0),
			     (4))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (0), (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1),
			     (0))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (1), (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1),
			     (1))] =
			    1.0 +
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (1),
				  (1))] + tmp1 * 2.0 * dx2;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1),
			     (2))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (1), (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1),
			     (3))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (1), (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1),
			     (4))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (1), (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2),
			     (0))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (2), (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2),
			     (1))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (2), (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2),
			     (2))] =
			    1.0 +
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (2),
				  (2))] + tmp1 * 2.0 * dx3;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2),
			     (3))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (2), (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2),
			     (4))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (2), (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3),
			     (0))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (3), (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3),
			     (1))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (3), (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3),
			     (2))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (3), (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3),
			     (3))] =
			    1.0 +
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (3),
				  (3))] + tmp1 * 2.0 * dx4;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3),
			     (4))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (3), (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4),
			     (0))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (4), (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4),
			     (1))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (4), (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4),
			     (2))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (4), (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4),
			     (3))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (4), (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4),
			     (4))] =
			    1.0 +
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i), (j), (k), (4),
				  (4))] + tmp1 * 2.0 * dx5;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0),
			     (0))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (0),
				  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (0),
								  (0))] -
			    tmp1 * dx1;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0),
			     (1))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (0),
				  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (0),
								  (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0),
			     (2))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (0),
				  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (0),
								  (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0),
			     (3))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (0),
				  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (0),
								  (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0),
			     (4))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (0),
				  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (0),
								  (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1),
			     (0))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (1),
				  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (1),
								  (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1),
			     (1))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (1),
				  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (1),
								  (1))] -
			    tmp1 * dx2;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1),
			     (2))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (1),
				  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (1),
								  (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1),
			     (3))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (1),
				  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (1),
								  (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1),
			     (4))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (1),
				  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (1),
								  (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2),
			     (0))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (2),
				  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (2),
								  (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2),
			     (1))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (2),
				  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (2),
								  (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2),
			     (2))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (2),
				  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (2),
								  (2))] -
			    tmp1 * dx3;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2),
			     (3))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (2),
				  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (2),
								  (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2),
			     (4))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (2),
				  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (2),
								  (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3),
			     (0))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (3),
				  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (3),
								  (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3),
			     (1))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (3),
				  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (3),
								  (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3),
			     (2))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (3),
				  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (3),
								  (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3),
			     (3))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (3),
				  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (3),
								  (3))] -
			    tmp1 * dx4;
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3),
			     (4))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (3),
				  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (3),
								  (4))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4),
			     (0))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (4),
				  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (4),
								  (0))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4),
			     (1))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (4),
				  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (4),
								  (1))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4),
			     (2))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (4),
				  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (4),
								  (2))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4),
			     (3))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (4),
				  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (4),
								  (3))];
			lhs[CALC_6D_IDX
			    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4),
			     (4))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (12, 12, 11, 5, 5, (i + 1), (j), (k), (4),
				  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5,
								  5, (i + 1),
								  (j), (k), (4),
								  (4))] -
			    tmp1 * dx5;
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1232 of bt.c
// The nested loops are swaped. 
__kernel void lhsy_0(double tmp1, __global double *u, double tmp2, double tmp3,
		     __global double *fjac, double c2, double c1,
		     __global double *njac, double c3c4, double con43,
		     double c1345, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		tmp1 = 1.0 / u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (0))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (1))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (2))] =
		    1.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (3))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (0))] =
		    -(u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))]) * tmp2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (1))] =
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] * tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (2))] =
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] * tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (3))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (0))] =
		    -(u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		      tmp2) +
		    0.50 * c2 *
		    ((u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		     tmp2);
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (1))] =
		    -c2 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
		    tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (2))] =
		    (2.0 -
		     c2) * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k),
					 (2))] * tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (3))] =
		    -c2 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
		    tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (4))] =
		    c2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (0))] =
		    -(u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) * tmp2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (1))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (2))] =
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] * tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (3))] =
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] * tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (0))] =
		    (c2 *
		     (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		     tmp2 -
		     c1 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] *
		     tmp1) * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k),
					   (2))] * tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (1))] =
		    -c2 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] * tmp2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (2))] =
		    c1 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] *
		    tmp1 -
		    0.50 * c2 *
		    ((u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		      3.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		     tmp2);
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (3))] =
		    -c2 * (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
			   u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		    tmp2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (4))] =
		    c1 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		    tmp1;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (0))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (1))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (2))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (3))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (4))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (0))] =
		    -c3c4 * tmp2 *
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (1))] =
		    c3c4 * tmp1;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (2))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (3))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (4))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (0))] =
		    -con43 * c3c4 * tmp2 *
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (1))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (2))] =
		    con43 * c3c4 * tmp1;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (3))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (4))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (0))] =
		    -c3c4 * tmp2 *
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (1))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (2))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (3))] =
		    c3c4 * tmp1;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (4))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (0))] =
		    -(c3c4 -
		      c1345) * tmp3 *
		    (((u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))]) *
		      (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))]))) -
		    (con43 * c3c4 -
		     c1345) * tmp3 *
		    (((u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))]) *
		      (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))]))) -
		    (c3c4 -
		     c1345) * tmp3 *
		    (((u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		      (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]))) -
		    c1345 * tmp2 *
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (1))] =
		    (c3c4 -
		     c1345) * tmp2 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k),
						   (1))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (2))] =
		    (con43 * c3c4 -
		     c1345) * tmp2 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k),
						   (2))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (3))] =
		    (c3c4 -
		     c1345) * tmp2 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k),
						   (3))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (4))] =
		    (c1345) * tmp1;
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1336 of bt.c
// The nested loops are swaped. 
__kernel void lhsy_1(double tmp1, double dt, double ty1, double tmp2,
		     double ty2, __global double *lhs, __global double *fjac,
		     __global double *njac, double dy1, double dy2, double dy3,
		     double dy4, double dy5, int __ocl_k_bound,
		     int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		tmp1 = dt * ty1;
		tmp2 = dt * ty2;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (0),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (0),
							  (0))] - tmp1 * dy1;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (0),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (0),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (0),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (0),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (0),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (0),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (0),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (0),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (1),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (1),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (1),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (1),
							  (1))] - tmp1 * dy2;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (1),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (1),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (1),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (1),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (1),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (1),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (2),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (2),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (2),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (2),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (2),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (2),
							  (2))] - tmp1 * dy3;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (2),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (2),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (2),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (2),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (3),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (3),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (3),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (3),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (3),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (3),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (3),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (3),
							  (3))] - tmp1 * dy4;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (3),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (3),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (4),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (4),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (4),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (4),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (4),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (4),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (4),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (4),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j - 1), (k), (4),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j - 1), (k), (4),
							  (4))] - tmp1 * dy5;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0), (0))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (0))]
		    + tmp1 * 2.0 * dy1;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (0), (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (0), (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (0), (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (0), (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (1), (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1), (1))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (1))]
		    + tmp1 * 2.0 * dy2;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (1), (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (1), (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (1), (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (2), (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (2), (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2), (2))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (2))]
		    + tmp1 * 2.0 * dy3;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (2), (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (2), (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (3), (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (3), (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (3), (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3), (3))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (3))]
		    + tmp1 * 2.0 * dy4;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (3), (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (4), (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (4), (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (4), (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (4), (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4), (4))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (4))]
		    + tmp1 * 2.0 * dy5;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (0),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (0),
							  (0))] - tmp1 * dy1;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (0),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (0),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (0),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (0),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (0),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (0),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (0),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (0),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (1),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (1),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (1),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (1),
							  (1))] - tmp1 * dy2;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (1),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (1),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (1),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (1),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (1),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (1),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (2),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (2),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (2),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (2),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (2),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (2),
							  (2))] - tmp1 * dy3;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (2),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (2),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (2),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (2),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (3),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (3),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (3),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (3),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (3),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (3),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (3),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (3),
							  (3))] - tmp1 * dy4;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (3),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (3),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (4),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (4),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (4),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (4),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (4),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (4),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (4),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (4),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j + 1), (k), (4),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j + 1), (k), (4),
							  (4))] - tmp1 * dy5;
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1530 of bt.c
// The nested loops are swaped. 
__kernel void lhsz_0(double tmp1, __global double *u, double tmp2, double tmp3,
		     __global double *fjac, double c2, double c1,
		     __global double *njac, double c3c4, double con43,
		     double c3, double c4, double c1345, int __ocl_k_bound,
		     int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		tmp1 = 1.0 / u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (0))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (1))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (2))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (3))] =
		    1.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (0))] =
		    -(u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) * tmp2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (1))] =
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] * tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (2))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (3))] =
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] * tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (0))] =
		    -(u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) * tmp2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (1))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (2))] =
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] * tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (3))] =
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] * tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (0))] =
		    -(u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
		      tmp2) +
		    0.50 * c2 *
		    ((u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		     tmp2);
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (1))] =
		    -c2 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
		    tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (2))] =
		    -c2 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		    tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (3))] =
		    (2.0 -
		     c2) * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k),
					 (3))] * tmp1;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (4))] =
		    c2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (0))] =
		    (c2 *
		     (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		     tmp2 -
		     c1 * (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] *
			   tmp1)) * (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k),
						   (3))] * tmp1);
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (1))] =
		    -c2 * (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
			   u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		    tmp2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (2))] =
		    -c2 * (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
			   u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		    tmp2;
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (3))] =
		    c1 * (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] *
			  tmp1) -
		    0.50 * c2 *
		    ((u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		      3.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		     tmp2);
		fjac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (4))] =
		    c1 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
		    tmp1;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (0))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (1))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (2))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (3))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (4))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (0))] =
		    -c3c4 * tmp2 *
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (1))] =
		    c3c4 * tmp1;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (2))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (3))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (4))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (0))] =
		    -c3c4 * tmp2 *
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (1))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (2))] =
		    c3c4 * tmp1;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (3))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (4))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (0))] =
		    -con43 * c3c4 * tmp2 *
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (1))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (2))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (3))] =
		    con43 * c3 * c4 * tmp1;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (4))] =
		    0.0;
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (0))] =
		    -(c3c4 -
		      c1345) * tmp3 *
		    (((u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))]) *
		      (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))]))) -
		    (c3c4 -
		     c1345) * tmp3 *
		    (((u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))]) *
		      (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))]))) -
		    (con43 * c3c4 -
		     c1345) * tmp3 *
		    (((u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		      (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]))) -
		    c1345 * tmp2 *
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (1))] =
		    (c3c4 -
		     c1345) * tmp2 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k),
						   (1))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (2))] =
		    (c3c4 -
		     c1345) * tmp2 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k),
						   (2))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (3))] =
		    (con43 * c3c4 -
		     c1345) * tmp2 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k),
						   (3))];
		njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (4))] =
		    (c1345) * tmp1;
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1634 of bt.c
// The nested loops are swaped. 
__kernel void lhsz_1(double tmp1, double dt, double tz1, double tmp2,
		     double tz2, __global double *lhs, __global double *fjac,
		     __global double *njac, double dz1, double dz2, double dz3,
		     double dz4, double dz5, int __ocl_k_bound,
		     int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		tmp1 = dt * tz1;
		tmp2 = dt * tz2;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (0),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (0),
							  (0))] - tmp1 * dz1;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (0),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (0),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (0),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (0),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (0),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (0),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (0), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (0),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (0),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (1),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (1),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (1),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (1),
							  (1))] - tmp1 * dz2;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (1),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (1),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (1),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (1),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (1), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (1),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (1),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (2),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (2),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (2),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (2),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (2),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (2),
							  (2))] - tmp1 * dz3;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (2),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (2),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (2), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (2),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (2),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (3),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (3),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (3),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (3),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (3),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (3),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (3),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (3),
							  (3))] - tmp1 * dz4;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (3), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (3),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (3),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (4),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (4),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (4),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (4),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (4),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (4),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (4),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (4),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (0), (4), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k - 1), (4),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k - 1), (4),
							  (4))] - tmp1 * dz5;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0), (0))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (0), (0))]
		    + tmp1 * 2.0 * dz1;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (0), (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (0), (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (0), (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (0), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (0), (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (1), (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1), (1))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (1), (1))]
		    + tmp1 * 2.0 * dz2;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (1), (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (1), (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (1), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (1), (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (2), (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (2), (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2), (2))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (2), (2))]
		    + tmp1 * 2.0 * dz3;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (2), (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (2), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (2), (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (3), (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (3), (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (3), (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3), (3))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (3), (3))]
		    + tmp1 * 2.0 * dz4;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (3), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (3), (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (4), (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (4), (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (4), (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k), (4), (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (1), (4), (4))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i), (j), (k), (4), (4))]
		    + tmp1 * 2.0 * dz5;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (0),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (0),
							  (0))] - tmp1 * dz1;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (0),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (0),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (0),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (0),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (0),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (0),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (0), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (0),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (0),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (1),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (1),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (1),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (1),
							  (1))] - tmp1 * dz2;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (1),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (1),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (1),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (1),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (1), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (1),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (1),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (2),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (2),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (2),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (2),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (2),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (2),
							  (2))] - tmp1 * dz3;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (2),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (2),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (2), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (2),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (2),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (3),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (3),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (3),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (3),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (3),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (3),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (3),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (3),
							  (3))] - tmp1 * dz4;
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (3), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (3),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (3),
							  (4))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (4),
			  (0))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (4),
							  (0))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (4),
			  (1))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (4),
							  (1))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (4),
			  (2))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (4),
							  (2))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (4),
			  (3))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (4),
							  (3))];
		lhs[CALC_6D_IDX
		    (13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (4), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (12, 12, 11, 5, 5, (i), (j), (k + 1), (4),
			  (4))] - tmp1 * njac[CALC_5D_IDX(12, 12, 11, 5, 5, (i),
							  (j), (k + 1), (4),
							  (4))] - tmp1 * dz5;
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1821 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_0(__global double *u, __global double *rho_i,
			    __global double *us, __global double *vs,
			    __global double *ws, __global double *square,
			    __global double *qs, int __ocl_k_bound,
			    int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double rho_inv;		/* Defined at bt.c : 1813 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rho_inv =
		    1.0 / u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))];
		rho_i[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] = rho_inv;
		us[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] =
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] * rho_inv;
		vs[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] =
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] * rho_inv;
		ws[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] =
		    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] * rho_inv;
		square[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] =
		    0.5 * (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] *
			   u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
			   u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] *
			   u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
			   u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] *
			   u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))]) *
		    rho_inv;
		qs[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] =
		    square[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] * rho_inv;
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1846 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_1(__global double *rhs, __global double *forcing,
			    int __ocl_k_bound, int __ocl_j_bound,
			    int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int m;			/* Defined at bt.c : 1812 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (m = 0; m < 5; m++) {
			rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
			    forcing[CALC_4D_IDX
				    (13, 13, 13, 6, (i), (j), (k), (m))];
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1862 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_2(__global double *us, __global double *rhs,
			    double dx1tx1, __global double *u, double tx2,
			    double dx2tx1, double xxcon2, double con43,
			    __global double *square, double c2, double dx3tx1,
			    __global double *vs, double dx4tx1,
			    __global double *ws, double dx5tx1, double xxcon3,
			    __global double *qs, double xxcon4, double xxcon5,
			    __global double *rho_i, double c1,
			    int __ocl_k_bound, int __ocl_j_bound,
			    int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double uijk;		/* Defined at bt.c : 1813 */
	double up1;		/* Defined at bt.c : 1813 */
	double um1;		/* Defined at bt.c : 1813 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		uijk = us[CALC_3D_IDX(13, 13, 13, (i), (j), (k))];
		up1 = us[CALC_3D_IDX(13, 13, 13, (i + 1), (j), (k))];
		um1 = us[CALC_3D_IDX(13, 13, 13, (i - 1), (j), (k))];
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))] +
		    dx1tx1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (0))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (0))]) -
		    tx2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (1))] -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (1))]);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		    dx2tx1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (1))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (1))]) +
		    xxcon2 * con43 * (up1 - 2.0 * uijk + um1) -
		    tx2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (1))] *
		     up1 -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (1))] *
		     um1 +
		     (u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (4))] -
		      square[CALC_3D_IDX(13, 13, 13, (i + 1), (j), (k))] -
		      u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (4))] +
		      square[CALC_3D_IDX(13, 13, 13, (i - 1), (j), (k))]) * c2);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		    dx3tx1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (2))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (2))]) +
		    xxcon2 * (vs[CALC_3D_IDX(13, 13, 13, (i + 1), (j), (k))] -
			      2.0 * vs[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
			      vs[CALC_3D_IDX(13, 13, 13, (i - 1), (j), (k))]) -
		    tx2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (2))] *
		     up1 -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (2))] *
		     um1);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] +
		    dx4tx1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (3))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (3))]) +
		    xxcon2 * (ws[CALC_3D_IDX(13, 13, 13, (i + 1), (j), (k))] -
			      2.0 * ws[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
			      ws[CALC_3D_IDX(13, 13, 13, (i - 1), (j), (k))]) -
		    tx2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (3))] *
		     up1 -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (3))] *
		     um1);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] +
		    dx5tx1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (4))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (4))]) +
		    xxcon3 * (qs[CALC_3D_IDX(13, 13, 13, (i + 1), (j), (k))] -
			      2.0 * qs[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
			      qs[CALC_3D_IDX(13, 13, 13, (i - 1), (j), (k))]) +
		    xxcon4 * (up1 * up1 - 2.0 * uijk * uijk + um1 * um1) +
		    xxcon5 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (4))] *
		     rho_i[CALC_3D_IDX(13, 13, 13, (i + 1), (j), (k))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] *
		     rho_i[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (4))] *
		     rho_i[CALC_3D_IDX(13, 13, 13, (i - 1), (j), (k))]) -
		    tx2 *
		    ((c1 *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (4))] -
		      c2 * square[CALC_3D_IDX(13, 13, 13, (i + 1), (j), (k))]) *
		     up1 -
		     (c1 *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (4))] -
		      c2 * square[CALC_3D_IDX(13, 13, 13, (i - 1), (j), (k))]) *
		     um1);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1925 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_3(__global double *rhs, int i, double dssp,
			    __global double *u, int __ocl_k_bound,
			    int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int k = get_global_id(1) + 1;
	int j = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp * (5.0 *
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (13, 13, 13, 5, (i + 1), (j), (k),
			       (m))] + u[CALC_4D_IDX(13, 13, 13, 5, (i + 2),
						     (j), (k), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1939 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_4(__global double *rhs, int i, double dssp,
			    __global double *u, int __ocl_k_bound,
			    int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int k = get_global_id(1) + 1;
	int j = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp * (-4.0 *
			    u[CALC_4D_IDX
			      (13, 13, 13, 5, (i - 1), (j), (k),
			       (m))] + 6.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i),
							   (j), (k),
							   (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (13, 13, 13, 5, (i + 1), (j), (k),
			       (m))] + u[CALC_4D_IDX(13, 13, 13, 5, (i + 2),
						     (j), (k), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1952 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_5(__global double *rhs, double dssp,
			    __global double *u, int __ocl_k_bound,
			    int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int m;			/* Defined at bt.c : 1812 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (m = 0; m < 5; m++) {
			rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))]
			    -
			    dssp *
			    (u
			     [CALC_4D_IDX
			      (13, 13, 13, 5, (i - 2), (j), (k),
			       (m))] - 4.0 * u[CALC_4D_IDX(13, 13, 13, 5,
							   (i - 1), (j), (k),
							   (m))] +
			     6.0 *
			     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
			     4.0 *
			     u[CALC_4D_IDX
			       (13, 13, 13, 5, (i + 1), (j), (k),
				(m))] + u[CALC_4D_IDX(13, 13, 13, 5, (i + 2),
						      (j), (k), (m))]);
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1969 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_6(__global double *rhs, int i, double dssp,
			    __global double *u, int __ocl_k_bound,
			    int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int k = get_global_id(1) + 1;
	int j = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i - 2), (j), (k), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (m))] +
		     6.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(13, 13, 13, 5, (i + 1), (j), (k), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1983 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_7(__global double *rhs, int i, double dssp,
			    __global double *u, int __ocl_k_bound,
			    int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int k = get_global_id(1) + 1;
	int j = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i - 2), (j), (k), (m))] -
		     4. *
		     u[CALC_4D_IDX(13, 13, 13, 5, (i - 1), (j), (k), (m))] +
		     5.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 1999 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_8(__global double *vs, __global double *rhs,
			    double dy1ty1, __global double *u, double ty2,
			    double dy2ty1, double yycon2, __global double *us,
			    double dy3ty1, double con43,
			    __global double *square, double c2, double dy4ty1,
			    __global double *ws, double dy5ty1, double yycon3,
			    __global double *qs, double yycon4, double yycon5,
			    __global double *rho_i, double c1,
			    int __ocl_k_bound, int __ocl_j_bound,
			    int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double vijk;		/* Defined at bt.c : 1813 */
	double vp1;		/* Defined at bt.c : 1813 */
	double vm1;		/* Defined at bt.c : 1813 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		vijk = vs[CALC_3D_IDX(13, 13, 13, (i), (j), (k))];
		vp1 = vs[CALC_3D_IDX(13, 13, 13, (i), (j + 1), (k))];
		vm1 = vs[CALC_3D_IDX(13, 13, 13, (i), (j - 1), (k))];
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))] +
		    dy1ty1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (0))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (0))]) -
		    ty2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (2))] -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (2))]);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		    dy2ty1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (1))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (1))]) +
		    yycon2 * (us[CALC_3D_IDX(13, 13, 13, (i), (j + 1), (k))] -
			      2.0 * us[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
			      us[CALC_3D_IDX(13, 13, 13, (i), (j - 1), (k))]) -
		    ty2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (1))] *
		     vp1 -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (1))] *
		     vm1);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		    dy3ty1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (2))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (2))]) +
		    yycon2 * con43 * (vp1 - 2.0 * vijk + vm1) -
		    ty2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (2))] *
		     vp1 -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (2))] *
		     vm1 +
		     (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (4))] -
		      square[CALC_3D_IDX(13, 13, 13, (i), (j + 1), (k))] -
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (4))] +
		      square[CALC_3D_IDX(13, 13, 13, (i), (j - 1), (k))]) * c2);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] +
		    dy4ty1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (3))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (3))]) +
		    yycon2 * (ws[CALC_3D_IDX(13, 13, 13, (i), (j + 1), (k))] -
			      2.0 * ws[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
			      ws[CALC_3D_IDX(13, 13, 13, (i), (j - 1), (k))]) -
		    ty2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (3))] *
		     vp1 -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (3))] *
		     vm1);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] +
		    dy5ty1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (4))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (4))]) +
		    yycon3 * (qs[CALC_3D_IDX(13, 13, 13, (i), (j + 1), (k))] -
			      2.0 * qs[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
			      qs[CALC_3D_IDX(13, 13, 13, (i), (j - 1), (k))]) +
		    yycon4 * (vp1 * vp1 - 2.0 * vijk * vijk + vm1 * vm1) +
		    yycon5 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (4))] *
		     rho_i[CALC_3D_IDX(13, 13, 13, (i), (j + 1), (k))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] *
		     rho_i[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (4))] *
		     rho_i[CALC_3D_IDX(13, 13, 13, (i), (j - 1), (k))]) -
		    ty2 *
		    ((c1 *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (4))] -
		      c2 * square[CALC_3D_IDX(13, 13, 13, (i), (j + 1), (k))]) *
		     vp1 -
		     (c1 *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (4))] -
		      c2 * square[CALC_3D_IDX(13, 13, 13, (i), (j - 1), (k))]) *
		     vm1);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2057 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_9(__global double *rhs, int j, double dssp,
			    __global double *u, int __ocl_k_bound,
			    int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int k = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp * (5.0 *
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (13, 13, 13, 5, (i), (j + 1), (k),
			       (m))] + u[CALC_4D_IDX(13, 13, 13, 5, (i),
						     (j + 2), (k), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2071 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_10(__global double *rhs, int j, double dssp,
			     __global double *u, int __ocl_k_bound,
			     int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int k = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp * (-4.0 *
			    u[CALC_4D_IDX
			      (13, 13, 13, 5, (i), (j - 1), (k),
			       (m))] + 6.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i),
							   (j), (k),
							   (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (13, 13, 13, 5, (i), (j + 1), (k),
			       (m))] + u[CALC_4D_IDX(13, 13, 13, 5, (i),
						     (j + 2), (k), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2084 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_11(__global double *rhs, double dssp,
			     __global double *u, int __ocl_k_bound,
			     int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int m;			/* Defined at bt.c : 1812 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (m = 0; m < 5; m++) {
			rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))]
			    -
			    dssp *
			    (u
			     [CALC_4D_IDX
			      (13, 13, 13, 5, (i), (j - 2), (k),
			       (m))] - 4.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i),
							   (j - 1), (k),
							   (m))] +
			     6.0 *
			     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
			     4.0 *
			     u[CALC_4D_IDX
			       (13, 13, 13, 5, (i), (j + 1), (k),
				(m))] + u[CALC_4D_IDX(13, 13, 13, 5, (i),
						      (j + 2), (k), (m))]);
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2101 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_12(__global double *rhs, int j, double dssp,
			     __global double *u, int __ocl_k_bound,
			     int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int k = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 2), (k), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (m))] +
		     6.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j + 1), (k), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2115 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_13(__global double *rhs, int j, double dssp,
			     __global double *u, int __ocl_k_bound,
			     int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int k = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 2), (k), (m))] -
		     4. *
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j - 1), (k), (m))] +
		     5. * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2131 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_14(__global double *ws, __global double *rhs,
			     double dz1tz1, __global double *u, double tz2,
			     double dz2tz1, double zzcon2, __global double *us,
			     double dz3tz1, __global double *vs, double dz4tz1,
			     double con43, __global double *square, double c2,
			     double dz5tz1, double zzcon3, __global double *qs,
			     double zzcon4, double zzcon5,
			     __global double *rho_i, double c1,
			     int __ocl_k_bound, int __ocl_j_bound,
			     int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	double wijk;		/* Defined at bt.c : 1813 */
	double wp1;		/* Defined at bt.c : 1813 */
	double wm1;		/* Defined at bt.c : 1813 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		wijk = ws[CALC_3D_IDX(13, 13, 13, (i), (j), (k))];
		wp1 = ws[CALC_3D_IDX(13, 13, 13, (i), (j), (k + 1))];
		wm1 = ws[CALC_3D_IDX(13, 13, 13, (i), (j), (k - 1))];
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))] +
		    dz1tz1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (0))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (0))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (0))]) -
		    tz2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (3))] -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (3))]);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		    dz2tz1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (1))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (1))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (1))]) +
		    zzcon2 * (us[CALC_3D_IDX(13, 13, 13, (i), (j), (k + 1))] -
			      2.0 * us[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
			      us[CALC_3D_IDX(13, 13, 13, (i), (j), (k - 1))]) -
		    tz2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (1))] *
		     wp1 -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (1))] *
		     wm1);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		    dz3tz1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (2))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (2))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (2))]) +
		    zzcon2 * (vs[CALC_3D_IDX(13, 13, 13, (i), (j), (k + 1))] -
			      2.0 * vs[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
			      vs[CALC_3D_IDX(13, 13, 13, (i), (j), (k - 1))]) -
		    tz2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (2))] *
		     wp1 -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (2))] *
		     wm1);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] +
		    dz4tz1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (3))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (3))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (3))]) +
		    zzcon2 * con43 * (wp1 - 2.0 * wijk + wm1) -
		    tz2 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (3))] *
		     wp1 -
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (3))] *
		     wm1 +
		     (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (4))] -
		      square[CALC_3D_IDX(13, 13, 13, (i), (j), (k + 1))] -
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (4))] +
		      square[CALC_3D_IDX(13, 13, 13, (i), (j), (k - 1))]) * c2);
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] +
		    dz5tz1 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (4))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (4))]) +
		    zzcon3 * (qs[CALC_3D_IDX(13, 13, 13, (i), (j), (k + 1))] -
			      2.0 * qs[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
			      qs[CALC_3D_IDX(13, 13, 13, (i), (j), (k - 1))]) +
		    zzcon4 * (wp1 * wp1 - 2.0 * wijk * wijk + wm1 * wm1) +
		    zzcon5 *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (4))] *
		     rho_i[CALC_3D_IDX(13, 13, 13, (i), (j), (k + 1))] -
		     2.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (4))] *
		     rho_i[CALC_3D_IDX(13, 13, 13, (i), (j), (k))] +
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (4))] *
		     rho_i[CALC_3D_IDX(13, 13, 13, (i), (j), (k - 1))]) -
		    tz2 *
		    ((c1 *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (4))] -
		      c2 * square[CALC_3D_IDX(13, 13, 13, (i), (j), (k + 1))]) *
		     wp1 -
		     (c1 *
		      u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (4))] -
		      c2 * square[CALC_3D_IDX(13, 13, 13, (i), (j), (k - 1))]) *
		     wm1);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2190 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_15(__global double *rhs, int k, double dssp,
			     __global double *u, int __ocl_j_bound,
			     int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp * (5.0 *
			    u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (13, 13, 13, 5, (i), (j), (k + 1),
			       (m))] + u[CALC_4D_IDX(13, 13, 13, 5, (i), (j),
						     (k + 2), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2204 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_16(__global double *rhs, int k, double dssp,
			     __global double *u, int __ocl_j_bound,
			     int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp * (-4.0 *
			    u[CALC_4D_IDX
			      (13, 13, 13, 5, (i), (j), (k - 1),
			       (m))] + 6.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i),
							   (j), (k),
							   (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (13, 13, 13, 5, (i), (j), (k + 1),
			       (m))] + u[CALC_4D_IDX(13, 13, 13, 5, (i), (j),
						     (k + 2), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2217 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_17(__global double *rhs, double dssp,
			     __global double *u, int __ocl_k_bound,
			     int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
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
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int m;			/* Defined at bt.c : 1812 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (m = 0; m < 5; m++) {
			rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))]
			    -
			    dssp *
			    (u
			     [CALC_4D_IDX
			      (13, 13, 13, 5, (i), (j), (k - 2),
			       (m))] - 4.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i),
							   (j), (k - 1),
							   (m))] +
			     6.0 *
			     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
			     4.0 *
			     u[CALC_4D_IDX
			       (13, 13, 13, 5, (i), (j), (k + 1),
				(m))] + u[CALC_4D_IDX(13, 13, 13, 5, (i), (j),
						      (k + 2), (m))]);
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2234 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_18(__global double *rhs, int k, double dssp,
			     __global double *u, int __ocl_j_bound,
			     int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 2), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (m))] +
		     6.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k + 1), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2248 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_19(__global double *rhs, int k, double dssp,
			     __global double *u, int __ocl_j_bound,
			     int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 2), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k - 1), (m))] +
		     5.0 * u[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))]);
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2261 of bt.c
// The nested loops are swaped. 
__kernel void compute_rhs_20(__global int *grid_points, __global double *rhs,
			     double dt, int __ocl_k_bound, int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int k = get_global_id(1) + 1;
	int j = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int i;			/* Defined at bt.c : 1812 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (i = 1; i < grid_points[(0)] - 1; i++) {
			rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))]
			    * dt;
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2787 of bt.c
// The nested loops are swaped. 
__kernel void x_backsubstitute_0(__global double *rhs, int i,
				 __global double *lhs, int __ocl_k_bound,
				 int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int k = get_global_id(1) + 1;
	int j = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int n;			/* Defined at bt.c : 2783 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (n = 0; n < 5; n++) {
			rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))]
			    -
			    lhs[CALC_6D_IDX
				(13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (m),
				 (n))] * rhs[CALC_4D_IDX(13, 13, 13, 5, (i + 1),
							 (j), (k), (n))];
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2825 of bt.c
// The nested loops are swaped. 
__kernel void x_solve_cell_0(__global double *lhs, __global double *rhs,
			     int __ocl_k_bound, int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((13 * 13 * 3 * 5 * 5) * (0)) +
				   ((13 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
				  (((13 * 13 * 3 * 5 * 5) * (0)) +
				   ((13 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				  (((13 * 13 * 5) * (0)) + ((13 * 5) * (j)) +
				   ((5) * (k))));
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2846 of bt.c
// The nested loops are swaped. 
__kernel void x_solve_cell_1(__global double *lhs, int i, __global double *rhs,
			     int __ocl_k_bound, int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((13 * 13 * 5) * (i - 1)) +
				     ((13 * 5) * (j)) + ((5) * (k))),
				    (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
				     ((5) * (k))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((13 * 13 * 3 * 5 * 5) * (i - 1)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))));
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((13 * 13 * 3 * 5 * 5) * (i)) +
				   ((13 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
				  (((13 * 13 * 3 * 5 * 5) * (i)) +
				   ((13 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				  (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
				   ((5) * (k))));
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 2878 of bt.c
// The nested loops are swaped. 
__kernel void x_solve_cell_2(__global double *lhs, int isize,
			     __global double *rhs, int i, int __ocl_k_bound,
			     int __ocl_j_bound)
{
	// Declare index variables (BEGIN)
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((13 * 13 * 3 * 5 * 5) * (isize)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((13 * 13 * 5) * (isize - 1)) +
				     ((13 * 5) * (j)) + ((5) * (k))),
				    (((13 * 13 * 5) * (isize)) +
				     ((13 * 5) * (j)) + ((5) * (k))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((13 * 13 * 3 * 5 * 5) * (isize)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((13 * 13 * 3 * 5 * 5) * (isize - 1)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				    (((13 * 13 * 3 * 5 * 5) * (isize)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))));
		binvrhs_p0_p1(lhs, rhs,
			      (((13 * 13 * 3 * 5 * 5) * (i)) +
			       ((13 * 3 * 5 * 5) * (j)) + ((3 * 5 * 5) * (k)) +
			       ((5 * 5) * (1))),
			      (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
			       ((5) * (k))));
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 3434 of bt.c
// The nested loops are swaped. 
__kernel void y_backsubstitute_0(__global double *rhs, int j,
				 __global double *lhs, int __ocl_k_bound,
				 int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int m = get_global_id(0);
	int k = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
	if (!(m < 5)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int n;			/* Defined at bt.c : 3429 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (n = 0; n < 5; n++) {
			rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(13, 13, 13, 5, (i), (j), (k), (m))]
			    -
			    lhs[CALC_6D_IDX
				(13, 13, 13, 3, 5, 5, (i), (j), (k), (2), (m),
				 (n))] * rhs[CALC_4D_IDX(13, 13, 13, 5, (i),
							 (j + 1), (k), (n))];
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 3472 of bt.c
// The nested loops are swaped. 
__kernel void y_solve_cell_0(__global double *lhs, __global double *rhs,
			     int __ocl_k_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((13 * 13 * 3 * 5 * 5) * (i)) +
				   ((13 * 3 * 5 * 5) * (0)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
				  (((13 * 13 * 3 * 5 * 5) * (i)) +
				   ((13 * 3 * 5 * 5) * (0)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				  (((13 * 13 * 5) * (i)) + ((13 * 5) * (0)) +
				   ((5) * (k))));
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 3493 of bt.c
// The nested loops are swaped. 
__kernel void y_solve_cell_1(__global double *lhs, int j, __global double *rhs,
			     int __ocl_k_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((13 * 13 * 5) * (i)) +
				     ((13 * 5) * (j - 1)) + ((5) * (k))),
				    (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
				     ((5) * (k))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j - 1)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))));
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((13 * 13 * 3 * 5 * 5) * (i)) +
				   ((13 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
				  (((13 * 13 * 3 * 5 * 5) * (i)) +
				   ((13 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				  (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
				   ((5) * (k))));
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 3526 of bt.c
// The nested loops are swaped. 
__kernel void y_solve_cell_2(__global double *lhs, int jsize,
			     __global double *rhs, int __ocl_k_bound,
			     int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (jsize)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((13 * 13 * 5) * (i)) +
				     ((13 * 5) * (jsize - 1)) + ((5) * (k))),
				    (((13 * 13 * 5) * (i)) +
				     ((13 * 5) * (jsize)) + ((5) * (k))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (jsize)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (jsize - 1)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (jsize)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))));
		binvrhs_p0_p1(lhs, rhs,
			      (((13 * 13 * 3 * 5 * 5) * (i)) +
			       ((13 * 3 * 5 * 5) * (jsize)) +
			       ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
			      (((13 * 13 * 5) * (i)) + ((13 * 5) * (jsize)) +
			       ((5) * (k))));
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 3598 of bt.c
// The nested loops are swaped. 
__kernel void z_backsubstitute_0(__global double *rhs, __global double *lhs,
				 int __ocl_k_bound, int __ocl_j_bound,
				 int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int k = get_global_id(0);
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	int m;			/* Defined at bt.c : 3594 */
	int n;			/* Defined at bt.c : 3594 */
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		for (m = 0; m < 5; m++) {
			for (n = 0; n < 5; n++) {
				rhs[CALC_4D_IDX
				    (13, 13, 13, 5, (i), (j), (k), (m))] =
				    rhs[CALC_4D_IDX
					(13, 13, 13, 5, (i), (j), (k),
					 (m))] - lhs[CALC_6D_IDX(13, 13, 13, 3,
								 5, 5, (i), (j),
								 (k), (2), (m),
								 (n))] *
				    rhs[CALC_4D_IDX
					(13, 13, 13, 5, (i), (j), (k + 1),
					 (n))];
			}
		}
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 3640 of bt.c
// The nested loops are swaped. 
__kernel void z_solve_cell_0(__global double *lhs, __global double *rhs,
			     int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((13 * 13 * 3 * 5 * 5) * (i)) +
				   ((13 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (0)) + ((5 * 5) * (1))),
				  (((13 * 13 * 3 * 5 * 5) * (i)) +
				   ((13 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (0)) + ((5 * 5) * (2))),
				  (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
				   ((5) * (0))));
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 3662 of bt.c
// The nested loops are swaped. 
__kernel void z_solve_cell_1(__global double *lhs, int k, __global double *rhs,
			     int __ocl_j_bound, int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
				     ((5) * (k - 1))),
				    (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
				     ((5) * (k))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k - 1)) + ((5 * 5) * (2))),
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))));
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((13 * 13 * 3 * 5 * 5) * (i)) +
				   ((13 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
				  (((13 * 13 * 3 * 5 * 5) * (i)) +
				   ((13 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				  (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
				   ((5) * (k))));
	}
// OpenCL kernel (END)

}

// The original loop is defined at line: 3699 of bt.c
// The nested loops are swaped. 
__kernel void z_solve_cell_2(__global double *lhs, int ksize,
			     __global double *rhs, int __ocl_j_bound,
			     int __ocl_i_bound)
{
	// Declare index variables (BEGIN)
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	// Declare index variables (END)
	// Declare private variables (BEGIN)
	// Declare private variables (END)

//COPYIN (START)
//COPYIN (END)

// OpenCL kernel (BEGIN)
	{
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (ksize)) + ((5 * 5) * (0))),
				    (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
				     ((5) * (ksize - 1))),
				    (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
				     ((5) * (ksize))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (ksize)) + ((5 * 5) * (0))),
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (ksize - 1)) +
				     ((5 * 5) * (2))),
				    (((13 * 13 * 3 * 5 * 5) * (i)) +
				     ((13 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (ksize)) +
				     ((5 * 5) * (1))));
		binvrhs_p0_p1(lhs, rhs,
			      (((13 * 13 * 3 * 5 * 5) * (i)) +
			       ((13 * 3 * 5 * 5) * (j)) +
			       ((3 * 5 * 5) * (ksize)) + ((5 * 5) * (1))),
			      (((13 * 13 * 5) * (i)) + ((13 * 5) * (j)) +
			       ((5) * (ksize))));
	}
// OpenCL kernel (END)

}
