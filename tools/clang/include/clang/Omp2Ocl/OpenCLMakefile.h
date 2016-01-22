#ifndef __OPENCL_MAKEFILE_H__
#define __OPENCL_MAKEFILE_H__
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include<vector>
#include<string>

using namespace std;

namespace clang
{
	class OpenCLMakefile
	{
		string hostF;
		string kernelF;
		string errorFile;
		llvm::raw_fd_ostream Out;

		public:
			OpenCLMakefile(string hostFile, string kernelFile)
				: Out("Makefile.ocl", errorFile)
			{
				hostF = hostFile;
				kernelF = kernelFile;
			}

			void doIt();

			~OpenCLMakefile()
			{
				Out.close();
			}
	};
}

#endif
