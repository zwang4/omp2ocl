#ifndef __OPENCL_LOOP_VECTORIZATION_H__
#define __OPENCL_LOOP_VECTORIZATION_H__
/*!
 * This function performs simple code vectorisation
 *
 *
 */

#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include "clang/Omp2Ocl/OpenCLCodeOptimisation.h" 

using namespace std;
using namespace clang;

namespace clang
{
	class OpenCLVectorisation : public OpenCLCodeOptimisation
	{
		public:
			OpenCLVectorisation(ASTContext& CTX, OpenCLKernelLoop* l)
			: OpenCLCodeOptimisation(CTX, l)
			{
				
			}

			void doIt();

	};
}

#endif

