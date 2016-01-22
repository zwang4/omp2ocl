#ifndef __OPENCLLOOPSCAN_H__
#define __OPENCLLOOPSCAN_H__

#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include <vector>

using namespace std;
using namespace clang;

namespace clang
{
	class OpenCLLoopScan
	{
		ASTContext& Context;
		vector<OpenCLKernelLoop*> oclLoops;
		vector<FunctionDecl*> revisedFuncs;
		static vector<QualType> qtypes;

		TypedefDecl* getTypeDefRef(string ty);
		void reviseCalledArgs(OpenCLKernelLoop* loop);
		//void scanThreadPrivateVars(OpenCLKernelLoop* loop);
		void scanLoop(OpenCLKernelLoop* loop);
		void collectGlobalInputArguments(OpenCLKernelLoop* loop);
		void scanNonPrimitiveType(OpenCLKernelLoop* loop, vector<DeclRefExpr*>& decls);
		void retriveOpenCLNDRangeVar(OpenCLKernelLoop* loop, ForStmt* Node, unsigned int orig_index);
		void retriveOpenCLNDRangeVars(OpenCLKernelLoop* loop);

		public:
		OpenCLLoopScan(ASTContext& Ctx, vector<OpenCLKernelLoop*>& oclLs)
			: Context(Ctx), oclLoops(oclLs)
		{

		}

		void _do();
		virtual void doIt();
		void addRevisedFunc(FunctionDecl* D);

		vector<FunctionDecl*>& getRevisedFuncs();
		vector<OpenCLKernelLoop*>& getOclLoops();
		ASTContext& getContext() { return Context; }
	};
}

#endif

