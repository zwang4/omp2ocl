#ifndef __OPENCLKERNELNAME_H__
#define __OPENCLKERNELNAME_H__

#include "clang/AST/Stmt.h"
#include <vector>
#include <string>

using namespace std;
using namespace clang;

namespace clang
{
	class KernelIndex {
		public:
			KernelIndex(string func)
			{
				function = func;
				index=1;
			}
			string function;
			unsigned int index;
	};

	class OpenCLKernelName {
		static vector<KernelIndex> oclKernelIdxs;
		public:
			static string getOpenCLKernelName(FunctionDecl* f);
			static unsigned int getOCLKernelIdx(string funcName);
	};
}


#endif
