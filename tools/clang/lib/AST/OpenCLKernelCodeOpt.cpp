#include "clang/Omp2Ocl/OpenCLKernelCodeOpt.h"
#include "clang/Omp2Ocl/OpenCLLoopInterChange.h"
#include "clang/Omp2Ocl/OpenCLLocalMemOpt.h"
#include "clang/Omp2Ocl/OpenCLLoadSchedule.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"

#include <iostream>
using namespace std;

void OpenCLKernelCodeOpt::doIt()
{
	for (unsigned i=0; i<oclLoops.size(); i++)
	{
		OpenCLKernelLoop* loop = oclLoops[i];

		OpenCLLoopInterChange ocI(Context, loop);
		ocI.doIt();


		if (OCLCompilerOptions::EnableSoftwareCache)
		{
			//Software Cache Optimisation
			OpenCLLoadSchedule ocL(Context, loop);
			ocL.doIt();	
		}

		loop->setOptimised();
	}
}
