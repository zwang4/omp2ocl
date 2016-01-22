#ifndef __OPENCLCOMPLIEROPTIONS_H__
#define __OPENCLCOMPLIEROPTIONS_H__

#include "clang/AST/Decl.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace clang
{
	class OCLCompilerOptions
	{
		public:
			static bool EnableSoftwareCache;
			static bool EnableDebugCG;
			static bool UseLocalMemory;
			static unsigned DefaultParallelDepth;
			static bool UserDefParallelDepth;
			static bool EnableLoopInterchange;
			static bool EnableMLFeatureCollection;
			static bool UseArrayLinearize;
			static bool EnableGPUTLs;
			static bool StrictTLSChecking;
			static bool TLSCheckAtProgramEnd;
			static bool printLinearMacros;
			static bool GenProfilingFunc;
			static bool OclTLSMechanism;
			static void printCompilerOptions();
			static void commentCompilerOptions(llvm::raw_fd_ostream& out);
	};

	class OpenCLKernelLoop;

	class OCLCompilerOptionAction
	{
		public:
			static bool isLocalVar(ValueDecl* D);
			static bool isLocalVar(OpenCLKernelLoop* loop, ValueDecl* D);
	};
}

#endif
