#ifndef __OPENCLLOOPINVMOV_H__ 
#define __OPENCLLOOPINVMOV_H__

#include <iostream>
#include <vector>

#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLCodeOptimisation.h"

using namespace std;
using namespace clang;

namespace clang
{
	class OpenCLLoopInvarMov : public OpenCLCodeOptimisation
	{
		public:
			OpenCLLoopInvarMov(ASTContext& C, OpenCLKernelLoop* l)
			: OpenCLCodeOptimisation(C, l)
			{
				
			}
	};
}

#endif
