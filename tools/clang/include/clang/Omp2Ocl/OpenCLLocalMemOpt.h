//This file prefetch a global memory buffer and
//store it to a local memory buffer
//

#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include "clang/Omp2Ocl/OpenCLCodeOptimisation.h"
#include <vector>

using namespace clang;
using namespace std;

namespace clang {

	class OpenCLLocalMemOpt
	{
		ASTContext& Context;
		public:
			OpenCLLocalMemOpt(ASTContext& Ctx) : Context(Ctx)
			{
				
			}	

			static bool isLocalVar(ValueDecl* d);
			static bool isLocalVar(OpenCLKernelLoop* loop, ValueDecl* d);
			void printLocalVars();
			static void genPreloadCode(llvm::raw_ostream& Out, ValueDecl* d, string passInName);		
			static void declareLocalVar(llvm::raw_ostream& Out, ValueDecl* d);		
	};

}
