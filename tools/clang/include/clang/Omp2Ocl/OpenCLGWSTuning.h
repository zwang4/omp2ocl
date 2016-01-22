#ifndef __OPENCL_GWS_TUNING_H__
#define __OPENCL_GWS_TUNING_H__

#include <string>
#include <vector>
#include "llvm/Support/raw_ostream.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"

using namespace std;
using namespace clang;

namespace clang
{
	class OpenCLGWSTuning
	{
		string errorFile;
		llvm::raw_fd_ostream OS;
		vector<OpenCLKernelLoop*>& oclLoops;
		void addKernelMacros(string kernelName, unsigned i);
		public:
			OpenCLGWSTuning(vector<OpenCLKernelLoop*>& oclLs, string File) : OS(File.data(), errorFile), oclLoops(oclLs)
			{
				OS << "kernel_gws=[\n";
			}

			~OpenCLGWSTuning()
			{
				OS << "]\n";
				OS.close();
			}
			
			void doIt();
	};
}


#endif
