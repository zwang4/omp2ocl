#ifndef __OPENCLCOPYINROUTINE_H__
#define __OPENCLCOPYINROUTINE_H__

#include "clang/AST/StmtVisitor.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"

using namespace clang;

namespace clang
{
	class OpenCLCopyInRoutine
	{
		OpenCLKernelLoop* loop;
		llvm::raw_ostream& Out;

		void generateLocalCopyInCode(CopyInBuffer& buf);
		void generateGlobalCopyInCode(CopyInBuffer& buf);
		void declareMultiFactor();
		void declareAddOffset();

		public:
			OpenCLCopyInRoutine(OpenCLKernelLoop* Loop, llvm::raw_ostream& O)
				: loop(Loop), Out(O)
			{
			
			
			}

			static void genLocalCopyInCode(llvm::raw_ostream& Out, ValueDecl* d, string passInName, string type_prefix="");
			void declareCopyInBuffers();
			void doIt();
	};
}


#endif

