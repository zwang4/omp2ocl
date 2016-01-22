#ifndef __OPENCLCOLLECTCALLEDFUNCS_H__
#define __OPENCLCOLLECTCALLEDFUNCS_H__

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include "clang/Omp2Ocl/OpenCLLoopScan.h"
using namespace clang;

//This class collects functions that are used by the
//OpenCL Kernels

namespace clang
{
	class OpenCLCollectCalledFuncs : public OpenCLLoopScan
	{
		void scan(OpenCLKernelLoop* loop);
		void collectCandidateFuncs();
		void scanRenameFuncs();
		vector<FunctionDecl*> findExpendedFunc(vector<FunctionDecl*>& candidateFuncs);
		bool holistOpenCLNDRangeVarInFunction(FunctionDecl* D);
		void reviseFunctionWithOpenCLNDRangeVar(FunctionDecl* D, vector<DeclRefExpr*> globalVariables);
		bool findCall2GlobalBuffer(Stmt* Body, vector<OCLGlobalMemVar>& globalMemoryVariables,  vector<OMPThreadPrivateObject>& threadPrivates, vector<FunctionDecl*>& funcsNeed2Revised);
		void collectCallees();
		FunctionDecl* PickFuncDeclByName(vector<FunctionDecl*>& candidateFuncs, string name);

		public:
			OpenCLCollectCalledFuncs(ASTContext& Ctx, vector<OpenCLKernelLoop*> oclLs)
				: OpenCLLoopScan(Ctx, oclLs)
			{
			
			}

			virtual void doIt();
	};
}

#endif
