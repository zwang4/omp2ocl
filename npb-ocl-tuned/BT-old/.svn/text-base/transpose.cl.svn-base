#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

// 3D swap 2nd with 3rd dimension
__kernel void swap3D_d(__global double *odata, __global double *idata, int dim0, int dim1, int dim2, int blockDim, __local double* block)
{
    for (int j=0; j<dim0; j++)
    {
	    // read the matrix tile into shared memory
	    unsigned int xIndex = get_global_id(0);
	    unsigned int yIndex = get_global_id(1);

	    if((xIndex < dim2) && (yIndex < dim1))
	    {
	    	unsigned int index_in = j*dim1*dim2 + yIndex * dim2 + xIndex;
	    	block[get_local_id(1)*(blockDim+1)+get_local_id(0)] = idata[index_in];
	    }

	    barrier(CLK_LOCAL_MEM_FENCE);

	    // write the transposed matrix tile to global memory
	    xIndex = get_group_id(1) * blockDim + get_local_id(0);
	    yIndex = get_group_id(0) * blockDim + get_local_id(1);
	    if((xIndex < dim1) && (yIndex < dim2))
        {
	    	unsigned int index_out = j*dim1*dim2 + yIndex * dim1 + xIndex;
	    	odata[index_out] = block[get_local_id(0)*(blockDim+1)+get_local_id(1)];
	    }

	    barrier(CLK_LOCAL_MEM_FENCE);
    }
}

//__kernel void swap3D_d(__global double *odata, __global double *idata, int dim0, int dim1, int dim2, int blockDim, __local double* block)
//{
//    unsigned int xIndex = get_global_id(0);
//    unsigned int yIndex = get_global_id(1);
//
//    unsigned int xIndex2 = get_group_id(1) * blockDim + get_local_id(0);
//    unsigned int yIndex2 = get_group_id(0) * blockDim + get_local_id(1);
//
//    unsigned int offset = get_global_id(2);
//    unsigned int stride = get_global_size(2);
//
//    for (int j=offset; j<dim0; j+=stride)
//    {
//	    // read the matrix tile into shared memory
//	    if((xIndex < dim2) && (yIndex < dim1))
//	    {
//	    	unsigned int index_in = j*dim1*dim2 + yIndex * dim2 + xIndex;
//	    	block[get_local_id(1)*(blockDim+1)+get_local_id(0)] = idata[index_in];
//	    }
//
//	    barrier(CLK_LOCAL_MEM_FENCE);
//
//	    // write the transposed matrix tile to global memory
//	    if((xIndex2 < dim1) && (yIndex2 < dim2))
//        {
//	    	unsigned int index_out = j*dim1*dim2 + yIndex2 * dim1 + xIndex2;
//	    	odata[index_out] = block[get_local_id(0)*(blockDim+1)+get_local_id(1)];
//	    }
//
//	    barrier(CLK_LOCAL_MEM_FENCE);
//    }
//}

//__kernel void swap3D_d(__global double *odata, __global double *idata, int dim0, int dim1, int dim2, int blockDim)
//{
//    unsigned int xIndex = get_global_id(0);
//    unsigned int yIndex = get_global_id(1);
//
//    if((xIndex < dim2) && (yIndex < dim1))
//    {   
//      for (int j=0; j<dim0; j++)
//      {   
//          odata[j*dim1*dim2 + xIndex*dim2 + yIndex] = idata[j*dim1*dim2 + yIndex * dim2 + xIndex];
//      }   
//    }   
//}

//__kernel void swap3D_d(write_only image2d_t odata, read_only image2d_t idata, int dim0, int dim1, int dim2)
//{
//    unsigned int xIndex = get_global_id(0);
//    unsigned int yIndex = get_global_id(1);
//
//    sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;
//
//    if((xIndex < dim2) && (yIndex < dim1))
//    {
//      for (int j=0; j<dim0; j++)
//      {
//          unsigned int i = j*dim1*dim2 + yIndex*dim2 + xIndex;
//          float4 d = read_imagef (idata, sampler, (int2)(i%4225, i/4225));
//          i = j*dim1*dim2 + xIndex*dim2 + yIndex;
//          write_imagef (odata, (int2)(i%4225, i/4225), d);
//      }
//    }
//}



// 4D swap 2nd with 4th dimension
__kernel void swap4D_d(__global double *odata, __global double *idata, int dim0, int dim1, int dim2, int dim3, int blockDim, __local double* block)
{
    for (int j=0; j<dim0; j++)
    {
        for (int i=0; i<dim2; i++)
        {
	        // read the matrix tile into shared memory
	        unsigned int xIndex = get_global_id(0);
	        unsigned int yIndex = get_global_id(1);

	        if((xIndex < dim3) && (yIndex < dim1))
	        {
	        	unsigned int index_in = j*dim1* dim2*dim3 + i * dim3 + yIndex * dim2*dim3 + xIndex;
	        	block[get_local_id(1)*(blockDim+1)+get_local_id(0)] = idata[index_in];
	        }

	        barrier(CLK_LOCAL_MEM_FENCE);

	        // write the transposed matrix tile to global memory
	        xIndex = get_group_id(1) * blockDim + get_local_id(0);
	        yIndex = get_group_id(0) * blockDim + get_local_id(1);
	        if((xIndex < dim1) && (yIndex < dim3))
            {
	        	unsigned int index_out = j*dim1*dim2*dim3 + i*dim1 + yIndex * dim2*dim1 + xIndex;
	        	odata[index_out] = block[get_local_id(0)*(blockDim+1)+get_local_id(1)];
	        }

	        barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}



// 3D swap 2nd with 3rd dimension
__kernel void swap3D_f(__global float *odata, __global float *idata, int dim0, int dim1, int dim2, int blockDim, __local float* block)
{
    for (int j=0; j<dim0; j++)
    {
	    // read the matrix tile into shared memory
	    unsigned int xIndex = get_global_id(0);
	    unsigned int yIndex = get_global_id(1);

	    if((xIndex < dim2) && (yIndex < dim1))
	    {
	    	unsigned int index_in = j*dim1*dim2 + yIndex * dim2 + xIndex;
	    	block[get_local_id(1)*(blockDim+1)+get_local_id(0)] = idata[index_in];
	    }

	    barrier(CLK_LOCAL_MEM_FENCE);

	    // write the transposed matrix tile to global memory
	    xIndex = get_group_id(1) * blockDim + get_local_id(0);
	    yIndex = get_group_id(0) * blockDim + get_local_id(1);
	    if((xIndex < dim1) && (yIndex < dim2))
        {
	    	unsigned int index_out = j*dim1*dim2 + yIndex * dim1 + xIndex;
	    	odata[index_out] = block[get_local_id(0)*(blockDim+1)+get_local_id(1)];
	    }

	    barrier(CLK_LOCAL_MEM_FENCE);
    }
}


// 4D swap 2nd with 4th dimension
__kernel void swap4D_f(__global float *odata, __global float *idata, int dim0, int dim1, int dim2, int dim3, int blockDim, __local float* block)
{
    for (int j=0; j<dim0; j++)
    {
        for (int i=0; i<dim2; i++)
        {
	        // read the matrix tile into shared memory
	        unsigned int xIndex = get_global_id(0);
	        unsigned int yIndex = get_global_id(1);

	        if((xIndex < dim3) && (yIndex < dim1))
	        {
	        	unsigned int index_in = j*dim1* dim2*dim3 + i * dim3 + yIndex * dim2*dim3 + xIndex;
	        	block[get_local_id(1)*(blockDim+1)+get_local_id(0)] = idata[index_in];
	        }

	        barrier(CLK_LOCAL_MEM_FENCE);

	        // write the transposed matrix tile to global memory
	        xIndex = get_group_id(1) * blockDim + get_local_id(0);
	        yIndex = get_group_id(0) * blockDim + get_local_id(1);
	        if((xIndex < dim1) && (yIndex < dim3))
            {
	        	unsigned int index_out = j*dim1*dim2*dim3 + i*dim1 + yIndex * dim2*dim1 + xIndex;
	        	odata[index_out] = block[get_local_id(0)*(blockDim+1)+get_local_id(1)];
	        }

	        barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}



