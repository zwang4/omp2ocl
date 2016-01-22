#ifndef __OPENCLKERNELCODEOPT_H__
#define __OPENCLKERNELCODEOPT_H__
#include "clang/Omp2Ocl/OpenCLCodeOptimisation.h"
#include <vector>
using namespace clang;

namespace clang
{
	class OpenCLKernelCodeOpt
	{
		ASTContext& Context;
		vector<OpenCLKernelLoop*> oclLoops;
	
		public:
			OpenCLKernelCodeOpt(ASTContext& C, vector<OpenCLKernelLoop*> ls)
				: Context(C), oclLoops(ls)
			{
			
			}
		
			void doIt();
	};
}



#endif

