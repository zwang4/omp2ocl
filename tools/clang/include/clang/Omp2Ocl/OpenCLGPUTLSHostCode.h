#ifndef __OPENCLGPUTLSHOSTCODE_H__
#define __OPENCLGPUTLSHOSTCODE_H__

/*!
 * This schedules loads of __global variables so that
 * we can vectorise the load
 *
 *
 */

#include <iostream>
#include <vector>
#include <stack>
#include <list>
#include "clang/Frontend/FrontendActions.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/Support/Format.h"
#include <cstdio>
#include <iostream>
#include <vector>
#include "clang/Omp2Ocl/OpenCLGenericStmtVisitor.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"

using namespace clang;
using namespace std;

namespace clang{

	class GPUTLSLogContainer
	{
		public:
		ForStmt* Node;
		vector<ValueDecl*> _globalWriteBufs;
		vector<ValueDecl*> _globalLCWriteBuf;
		const FunctionDecl* FD;
		GPUTLSLogContainer(ForStmt* for_stmt, const FunctionDecl* d)
		{
			Node = for_stmt;
			_globalWriteBufs = for_stmt->getGlobalWriteBufs();
		       _globalLCWriteBuf = for_stmt->getGlobalLCWriteBufs();
		       FD = d;
		}
	};

	class GPUTLSLogContainerFD
	{
		public:
		const FunctionDecl* F;
		vector<ValueDecl*> _globalLCWriteBufs;
		vector<ValueDecl*> _globalWriteBufs;
		GPUTLSLogContainerFD(const FunctionDecl* FD)
		{
			this->F = FD;
		}
	}; 

	class OpenCLGPUTLSHostCode : public OpenCLGenericStmtVisitor {
		private:
			static vector<GPUTLSLogContainerFD> tls_FDC;
			static vector<GPUTLSLogContainer> tls_log_container;
			static void insertGPUTLsLog(ForStmt* for_stmt);
			static bool isInVector(vector<ValueDecl*>& vec, ValueDecl* v);
			static void genTLSKernelCall( llvm::raw_ostream &OS, vector<ValueDecl*>& _globalLCWriteBufs, vector<ValueDecl*>& _globalWriteBufs);
		public:
		OpenCLGPUTLSHostCode(ASTContext& Ctx) : 
			OpenCLGenericStmtVisitor(llvm::nulls(), Ctx, NULL, Ctx.PrintingPolicy, 0) 
		{
		}

		static vector<GPUTLSLogContainer>& getGPUTLsLogContainer()
		{
			return tls_log_container;
		}

		static vector<GPUTLSLogContainerFD>& getTLSFDC()
		{
			return tls_FDC;
		}
		
		static void genTLSCheckingKernelCodeAllFuncs(llvm::raw_ostream &OS);
		static void genTLSKernelCallLoopLevel(llvm::raw_ostream &OS, ForStmt* for_stmt);
		static void checkConflictFlag(llvm::raw_ostream &OS);
		static void printCheckingKernelHandles(llvm::raw_ostream &OS);
		static void buildCheckingKernelHandles(llvm::raw_ostream &OS);
		static void genTLSCheckingKernelCode(llvm::raw_ostream &O);
		virtual void VisitForStmt(ForStmt *Node);
	};
}

#endif
