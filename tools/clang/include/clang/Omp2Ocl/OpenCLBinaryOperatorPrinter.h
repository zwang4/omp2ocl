#ifndef __OPENCL_BINARYOPERATOR_PRINTER_H__
#define __OPENCL_BINARYOPERATOR_PRINTER_H__
#include <iostream>
#include <vector>
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"

using namespace std;
using namespace clang;

namespace clang
{
	class OpenCLBinaryOperatorPrinter
	{
		llvm::raw_ostream& OS;
		ASTContext& Context;
		bool ShouldPerformArithmTrans(BinaryOperator* Node);
		bool isGlobalThreadPrivateBuff(DeclRefExpr* e);

		public:
		OpenCLBinaryOperatorPrinter(llvm::raw_ostream &os, ASTContext& C): OS(os), Context(C)
		{
		}

		bool ShouldTransformBinaryOperator(BinaryOperator* Node);
	};
}

#endif
