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
#define GROUP_SIZE	128

//Functions that will be used by the kernels (START)


__kernel void rank_0_pre(__global int* buffer)
{
	int i = get_global_id(0);
	buffer[i] = 0;
}

//Functions that will be used by the kernels (END)

/* This is origined from a loop of is.c at line: 401 */
__kernel void
rank_0 (__global int * key_buff2, __global int * key_array,
	__global int * prv_buff1, unsigned int __ocl_i_bound)
{
  /*Global variables */
  unsigned int i = get_global_id (0);

  /* Private Variables */

  unsigned gsize_0 = get_global_size (0);
  unsigned gid_0 = get_global_id (0);

  unsigned __ocl_mult_factor = (gsize_0);
  unsigned __ocl_add_offset = ((gid_0));

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (; i<__ocl_i_bound; i+=gsize_0)
  {
    key_buff2[(i)] = key_array[i];
    int wid = key_buff2[i] *  __ocl_mult_factor + __ocl_add_offset;
    prv_buff1[wid] = prv_buff1[wid] + 1;
 }
  //OPENCL KERNEL END 
}

__kernel void rank_0_reduction(__global int* prv_buff1, __global int* reduction_buffer, int ocl_mult_factor)
{
	int j = get_global_id(0);

	int i;
	
	int sum = 0;
	for (i = 0; i<ocl_mult_factor; i++)
	//for (i = 0; i<1; i++)
	{
		sum = sum + reduction_buffer[j * ocl_mult_factor + i];
	}

	prv_buff1[j] = sum;
}
