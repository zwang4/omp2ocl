#ifndef __OPENCL_CODE_OPTIMISATION_H__
#define __OPENCL_CODE_OPTIMISATION_H__
#include "clang/AST/ASTContext.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"


namespace clang
{
	class OpenCLCodeOptimisation
	{
		ASTContext& Context;
		OpenCLKernelLoop* loop;
		public:
			OpenCLCodeOptimisation(ASTContext& C, OpenCLKernelLoop* l) 
				: Context(C), loop(l)
			{
			}
			
			OpenCLCodeOptimisation(ASTContext& C) : Context(C)
			{
				loop = NULL;
			}

			ASTContext& getContext()
			{
				return Context;
			}

			OpenCLKernelLoop* getLoop()
			{
				return loop;
			}

			virtual void doIt() = 0;
	};

}

#endif

